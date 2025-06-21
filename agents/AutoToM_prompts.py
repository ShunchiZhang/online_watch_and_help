import asyncio
import random
import time
from collections import Counter
from functools import partial

from json_repair import repair_json
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError

from utils.utils_exception import exception_info, is_openai_quota_exceeded
from utils.utils_graph import OBJECT_NAMES, TARGET_NAMES, TASK_NAMES
from utils.utils_logging import get_existing_logger_by_prefix


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
            if count > 0:
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

    def minus_objects(self, counter):
        self.objects = Object.from_counter(Object.to_counter(self.objects) - counter)

    def plus_objects(self, counter):
        self.objects = Object.from_counter(Object.to_counter(self.objects) + counter)

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

    def filter_low_conf(self, thres, min_num, normalize=True):
        self.particles = sorted(self.particles, key=lambda x: x.p, reverse=True)
        self.particles = self.particles[:min_num] + list(
            filter(lambda x: x.p >= thres, self.particles[min_num:])
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

    def minus_objects(self, counter):
        for particle in self.particles:
            particle.minus_objects(counter)

    def plus_objects(self, counter):
        for particle in self.particles:
            particle.plus_objects(counter)

    def probs_grab(self, in_log=False):
        probs = Counter()
        for particle in self.particles:
            for object in particle.objects:
                probs[object.type] += particle.p

        if in_log:
            for obj_type, prob in probs.items():
                probs[obj_type] = round(100 * prob, 1)

        return dict(probs.most_common())

    def probs_put(self, in_log=False):
        probs = Counter()
        for particle in self.particles:
            probs[particle.target.type] += particle.p

        if in_log:
            for obj_type, prob in probs.items():
                probs[obj_type] = round(100 * prob, 1)

        return dict(probs.most_common())

    def best_in_probs(self, probs):
        candidates = [x for x, p in probs.items() if p == max(probs.values())]
        if len(candidates) == 0:
            return None, 0
        else:
            answer = random.Random(0).choice(candidates)
            return answer, probs[answer]

    def best_grab(self):
        return self.best_in_probs(self.probs_grab(in_log=False))

    def best_put(self):
        return self.best_in_probs(self.probs_put(in_log=False))

    def __len__(self):
        return len(self.particles)


class Likelihood(BaseModel):
    likelihood: float = Field(
        ..., ge=0, le=1, description="likelihood in float number between 0 and 1."
    )


# * p(goal | next_human_action, curr_env_state, curr_human_state, key_action_history)
propose = """\
Human has been working on a task of moving some objects to a target location. The task type can only be one of the following: setting up a table, putting something in the dishwasher, putting something in the fridge, preparing food, or watching TV.

Your are a helpful assistant. In order to help human, please propose multiple hypotheses of [human's overall goal] (including both finished and potential future subgoals), base on the following information:

[current state]
{curr_env_state}

{curr_human_state}

[key action history]
{key_action_history}

[human's next action]
{next_human_action}

Hints:
- The task type is constant and the target location is unique, i.e., human will be consistently doing the same task (setting up a table, putting something in the dishwasher, putting something in the fridge, preparing food, or watching TV) and put all objects to the same location.
- Please propose diverse goals in both object type and count.

Output Requirements:
Please provide a probability distribution over n={n} hypotheses of [human's overall goal] (including both finished and potential future subgoals).
Your response should include the probability distribution formatted according to this JSON schema: {schema}
"""
propose = partial(propose.format, schema=GoalParticles.model_json_schema())

# * p(curr_human_action | goal, curr_env_state, curr_human_state, key_action_history)
forward_likelihood = """\
Human has been working on a task of moving some objects to a target location.

Please estimate the likelihood of: [human's next action] {next_human_action}

given the following information:

[current state]
{curr_env_state}

{curr_human_state}

[human's unfinished goals]
{goal}

Hints:
- If human holds nothing, human must grab a goal object, or walk towards a goal object or its room. **Please note: it is perfectly fine to grab any goal object, not necessarily the nearest one.**
- If human holds something, human must put the object to the target location, or walk towards the target location or its room.
- The action is impossible (p=0) if it contradicts the above rules.

Output Requirements:
Please provide a likelihood estimation in float number between 0 and 1.
Your response should include the likelihood estimation formatted according to this JSON schema: {schema}
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


async def call_gpt(aclient, prompt, model_slug, out_type, **kwargs):
    if model_slug.startswith("gpt"):
        base_args = dict(temperature=0)
    elif model_slug.startswith("o"):
        base_args = dict(reasoning_effort="high")
    else:
        raise ValueError(f"Invalid model_slug: {model_slug}")

    while True:
        try:
            resp = await aclient.chat.completions.create(
                model=model_slug,
                messages=[dict(role="user", content=prompt)],
                **{**base_args, **kwargs},
            )
            # * advantage of using repair_json instead of response_format:
            # * model can freely output CoT then final answer
            resp_text = resp.choices[0].message.content
            if out_type is not None:
                obj = repair_json(resp_text, return_objects=True)
                if isinstance(obj, list):
                    obj = obj[-1]  # keep the last valid json
                obj = out_type.model_validate(obj)
            else:
                obj = resp_text
            break
        except Exception as e:
            if is_openai_quota_exceeded(e):
                prefix = "CRITICAL ERROR"
            elif isinstance(e, (OpenAIError, ValidationError)):
                prefix = "HANDLED ERROR"
            else:
                prefix = "UNKNOWN ERROR"
            logger = get_existing_logger_by_prefix("main")
            logger.error(f"{prefix}: {exception_info(e)}")

            if not prefix.startswith("HANDLED"):
                raise e

    cost = Counter(
        dollar=sum(
            [
                resp.usage.prompt_tokens * LLM_PRICING[model_slug]["input"],
                resp.usage.completion_tokens * LLM_PRICING[model_slug]["output"],
            ]
        ),
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
    )

    return obj, cost


def call_gpt_batch(prompts, model_slug, out_type, **kwargs):
    async def _call_gpt_batch(prompts):
        async with AsyncOpenAI() as aclient:
            tasks = [
                call_gpt(aclient, prompt, model_slug, out_type, **kwargs)
                for prompt in prompts
            ]
            return await asyncio.gather(*tasks)

    t1 = time.time()
    results = asyncio.run(_call_gpt_batch(prompts))
    t2 = time.time()

    objs, costs = zip(*results)
    total_cost = sum(costs, Counter(time=t2 - t1))

    return objs, total_cost


if __name__ == "__main__":
    objs, cost = call_gpt_batch(
        [
            "19*19=?",
            "18*18=?",
            "17*17=?",
            "16*16=?",
            "15*15=?",
            "14*14=?",
            "13*13=?",
            "12*12=?",
            "11*11=?",
        ],
        model_slug="gpt-4o-mini",
        out_type=None,
    )
    print(cost)
    print(*objs, sep="\n")
