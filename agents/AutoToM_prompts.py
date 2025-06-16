import time
from collections import Counter
from functools import partial

from json_repair import repair_json
from openai import OpenAI
from pydantic import BaseModel, Field

from utils.utils_graph import OBJECT_NAMES, TARGET_NAMES, TASK_NAMES


class Object(BaseModel):
    type: str = Field(..., choices=OBJECT_NAMES)
    count: int = Field(..., ge=1)

    @staticmethod
    def to_counter(objects):
        counter = Counter()
        for obj in objects:
            counter[obj.type] += obj.count
        return counter

    @staticmethod
    def from_counter(counter):
        objects = []
        for obj_type, count in counter.items():
            objects.append(Object(type=obj_type, count=count))
        return objects


class Target(BaseModel):
    type: str = Field(..., choices=TARGET_NAMES)
    # preposition: str = Field(..., choices=["on", "inside"]) # * reduce planning space


class GoalParticle(BaseModel):
    task_name: str = Field(..., choices=TASK_NAMES)
    objects: list[Object] = Field(..., min_items=1)
    target: Target
    p: float = Field(..., ge=0, le=1, description="Probability of the goal proposal")

    def minus_objects(self, done_counter):
        left_counter = Object.to_counter(self.objects) - done_counter
        self.objects = Object.from_counter(left_counter)

    def to_natlang(self):
        counter = Object.to_counter(self.objects)
        objects = [f"{cnt} {name}" for name, cnt in counter.items()]
        return f"({self.task_name}) put {', '.join(objects)} to {self.target.type}"


class GoalParticles(BaseModel):
    particles: list[GoalParticle]

    def normalize(self):
        partition = sum(particle.p for particle in self.particles)
        for particle in self.particles:
            particle.p /= partition

    def reweight(self, probs, normalize=True):
        for particle, p in zip(self.particles, probs):
            particle.p *= p
        if normalize:
            self.normalize()

    def filter_low_conf(self, thres, normalize=True):
        self.particles = list(
            filter(lambda particle: particle.p >= thres, self.particles)
        )
        if normalize:
            self.normalize()

    def fill_particles(self, particles, max_particles, normalize=True):
        for particle in particles.particles:
            if particle.to_natlang() not in self.to_natlang().keys():
                self.particles.append(particle)

                if len(self.particles) == max_particles:
                    if normalize:
                        self.normalize()
                    break

    def to_natlang(self):
        contents = dict()
        for particle in self.particles:
            contents[particle.to_natlang()] = round(100 * particle.p, 1)
        return contents

    def minus_objects(self, done_counter):
        for particle in self.particles:
            particle.minus_objects(done_counter)

    def __len__(self):
        return len(self.particles)


class Likelihood(BaseModel):
    likelihood: float = Field(
        ..., ge=0, le=1, description="likelihood in float number between 0 and 1."
    )


# * p(goal | next_human_action, curr_env_state, curr_human_state, key_action_history)
propose = """\
Human has been searching for some objects to place them to a target location. It could be either setting up a table, putting something in the dishwasher, putting something in the fridge, preparing food, or watching TV.

Your are a helpful assistant. In order to help human, please propose multiple hypotheses of [human's goals to be completed], base on the following information:

[current state]
{curr_env_state}

{curr_human_state}

[key action history]
{key_action_history}

[human's next action]
{next_human_action}

Hints:
- The target location is unique, i.e., human will put all objects to the same location.
- In order to effectively help human, you should not only propose human's ongoing goals (e.g., human's currently grabbing something), but also novel goals that human has not grabbed yet.

Output Requirements:
Please provide a probability distribution over n={n} hypotheses of [human's goals to be completed].
Your response should include a JSON that follows the schema: {schema}
"""
propose = partial(propose.format, schema=GoalParticles.model_json_schema())

# * p(curr_human_action | goal, curr_env_state, curr_human_state, key_action_history)
forward_likelihood = """\
Human has been searching for some objects to place them to a target location. It could be either setting up a table, putting something in the dishwasher, putting something in the fridge, preparing food, or watching TV.

You are a logical reasoner about rational human behavior. Please estimate the likelihood of:

[human's next action]
{next_human_action}

given the following information:

[current state]
{curr_env_state}

{curr_human_state}

[key action history]
{key_action_history}

[human's goals to be completed]
{goal}

Hints:
- Human is rational, therefore, the likelihood of [human's next action] depends on how it contributes to [human's goals to be completed].
- An action is highly likely if it directly contributes to the goals, e.g.:
  - walking towards a goal object or its room, or
  - walking towards the target location or its room with a goal object, or
  - grabbing a goal object, or
  - putting a goal object to the target location.
- An action is unlikely if it does not contribute to the goals, e.g.:
  - grabbing an object not mentioned in the goals, or
  - placing an object somewhere other than the target location.

Output Requirements:
Please provide a likelihood estimation in float number between 0 and 1.
Your response should include a JSON that follows the schema: {schema}.
"""
forward_likelihood = partial(
    forward_likelihood.format, schema=Likelihood.model_json_schema()
)


LLM_PRICING = {
    # https://platform.openai.com/docs/pricing
    "gpt-4o": dict(input=2.50e-6, output=10.00e-6),
    "gpt-4o-mini": dict(input=0.15e-6, output=0.60e-6),
    "o3-mini": dict(input=1.10e-6, output=4.40e-6),
}


def call_gpt(prompt, llm_name):
    client = OpenAI()
    if llm_name.startswith("gpt"):
        kwargs = dict(temperature=0)
    elif llm_name.startswith("o"):
        kwargs = dict(reasoning_effort="high")
    else:
        raise ValueError(f"Invalid llm_name: {llm_name}")

    t1 = time.time()
    resp = client.chat.completions.create(
        model=llm_name,
        messages=[dict(role="user", content=prompt)],
        **kwargs,
    )
    t2 = time.time()

    obj = repair_json(resp.choices[0].message.content, return_objects=True)
    cost = dict(
        time=t2 - t1,
        dollar=sum(
            [
                resp.usage.prompt_tokens * LLM_PRICING[llm_name]["input"],
                resp.usage.completion_tokens * LLM_PRICING[llm_name]["output"],
            ]
        ),
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
    )

    return obj, cost
