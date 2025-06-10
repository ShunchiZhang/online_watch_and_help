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


question = """\
In the story below, human is searching for some objects to place them to a target location. It could be either setting up a table, putting something in the dishwasher, putting something in the fridge, preparing food, or watching TV.

Your are a helpful assistant. In order to help human finishing the task, please predict human's *overall* goal.

IMPORTANT INSTRUCTIONS:
- *overall* means you should **include both** completed subgoals and possible future subgoals.
- You should not stick on the completed subgoals, or *too obvious* future subgoals (if human currently grabs an object, predicting it as a future goal is *too obvious*), because your task is to **help** instead of accurate prediction. You can only achieve substantial helping by predicting something that human has not grabbed yet. However, while you focus on novel predictions, don't forget to include the completed subgoals.
- There is only a single **consistent** target location, i.e., human will put all objects to the same location. The object types and counts may vary though.
"""

propose = """\
## Story
{story}

{action}

## Output Format Requirements
Instead of simply giving one hypothesis of human's overall goal, please provide a probability distribution over n={n} overall goal hypotheses.
Your response should include a JSON that follows the schema: {schema}
"""
propose = partial(propose.format, schema=GoalParticles.model_json_schema())


forward_likelihood = """\
Based on the current environment state and human's action history andoverall goal (the overall goal could be partially completed or not started yet), what is the likelihood that human would take the current action described below?

Hints:
- An current action is highly likely if it directly contributes to any uncompleted subgoals, like:
  - walking toward a object (or its room) of an uncompleted subgoal, or
  - walking toward the target location (or its room) with a object of an uncompleted subgoal, or
  - grabbing a object of an uncompleted subgoal, or
  - putting a object of an uncompleted subgoal to the target location.
- An current action is unlikely if it involves:
  - grabbing an object not mentioned in the goal, or
  - grabbing an object but its related subgoals are all completed, or
  - placing an object somewhere other than the target location.

## Current Environment State
{story}

{state}

## Human's Action History
{action_history}

## Human's overall goal (could be partially completed or not started yet)
{particle}

## Current Action
{action}

What is the likelihood of the current action given the current environment state and human's action history and overall goal (the overall goal could be partially completed or not started yet), according to the hints above? Please check carefully if the current action maps to any of the hints above before giving your answer.

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
        kwargs = dict(temperature=0.1)
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
