import time
import traceback

from openai import OpenAI

from agents import AutoToM_prompts
from utils.utils_logging import get_my_logger


def call_gpt(prompt, llm_name):
    client = OpenAI()
    match llm_name:
        case "gpt-4o":
            input_per_token = 2.50e-6
            output_per_token = 10.00e-6
            kwargs = dict(temperature=0)
        case "o3-mini":
            input_per_token = 1.10e-6
            output_per_token = 4.40e-6
            kwargs = dict(reasoning_effort="high")
        case _:
            raise ValueError(f"Invalid llm_name: {llm_name}")

    t1 = time.time()
    completion = client.chat.completions.create(
        model=llm_name,
        messages=[dict(role="user", content=prompt)],
        **kwargs,
    )
    t2 = time.time()

    lg = get_my_logger()
    lg.debug(f"Time: {t2 - t1}s")

    cost = (
        completion.usage.prompt_tokens * input_per_token
        + completion.usage.completion_tokens * output_per_token
    )
    lg.debug(f"Cost: ${cost}")

    return completion.choices[0].message.content.strip()


class AutoToM:
    def __init__(
        self,
        conf_thres,
        filter_thres,
        num_particles,
        llm_name,
        method,
    ):
        """
        `conf_thres`: confidence threshold to act to help human
        `filter_thres`: filter threshold to filter out low-confidence hypos
        `num_particles`: number of particles to use
        `llm_name`: name of the llm to use, e.g., "gpt-4o" or "o3-mini"
        `method`: method to use, e.g., "autotom" or "llm"
        """
        self.llm_name = llm_name
        self.conf_thres = conf_thres
        self.filter_thres = filter_thres
        self.num_particles = num_particles
        self.method = method
        self.goal_hypos = dict()

    def update_done_goals(self, actions):
        self.done_objs = dict()
        self.done_tgt = None
        for action in actions:
            if "grab" in action:
                done_obj = action.split(" ")[-1]
                self.done_objs[done_obj] = self.done_objs.setdefault(done_obj, 0) + 1
            elif "put" in action:
                self.done_tgt = action.split(" ")[-1]
        self.saver.debug(f"[AutoToM] {self.done_objs = } | {self.done_tgt = }")

    def step(self, actions):
        self.update_done_goals(actions)
        prob_dist = self.get_prob_dist(actions)
        obj, tgt = self.parse_prob_dist(prob_dist)
        self.saver.debug(f"[AutoToM] {obj = } | {tgt = }")
        return obj, tgt

    def get_prob_dist(self, actions):
        """
        prob_dist example: {
            str(h[obj, tgt]): p,
            str(h[obj, tgt]): p,
        }
        """
        match self.method:
            case "autotom":
                prob_dist = self.get_prob_dist_autotom(actions)
            case "llm":
                prob_dist = self.propose_goal(actions, n=self.num_particles)
            case _:
                raise ValueError(f"Invalid method: {self.method}")
        self.saver.debug(f"[particles] {prob_dist = }")
        return prob_dist

    def parse_prob_dist(self, prob_dist):
        if len(prob_dist) == 0:
            return None, None
        obj_probs = dict()
        tgt_probs = dict()
        for hypo, prob in prob_dist.items():
            obj_list = eval(hypo)["objects"]
            for name in obj_list:
                # * handle done objects
                if name in self.done_objs:
                    # * skip fully done objects
                    if self.done_objs[name] >= self.done_cnt:
                        continue
                    # * execute partially done objects
                    else:
                        obj_probs[name] = 1
                else:
                    obj_probs[name] = obj_probs.setdefault(name, 0) + prob

            if self.done_tgt is not None:
                # * fix the target if human confirmed
                tgt_probs[self.done_tgt] = 1
            else:
                tgt = eval(hypo)["target"]
                tgt_probs[tgt] = tgt_probs.setdefault(tgt, 0) + prob
        obj, obj_prob = max(obj_probs.items(), key=lambda x: x[1])
        tgt, tgt_prob = max(tgt_probs.items(), key=lambda x: x[1])
        self.saver.debug(f"[AutoToM] {obj_probs = }")
        self.saver.debug(f"[AutoToM] {tgt_probs = }")
        return (
            obj if obj_prob >= self.conf_thres["grab"] else None,
            tgt if tgt_prob >= self.conf_thres["put"] else None,
        )

    def propose_goal(self, actions, n):
        """
        ! IMPORTANT: maintain overall goal as particles, instead of future goals.

        return {
            str(h[obj, tgt]): p,
            str(h[obj, tgt]): p,
        }
        """
        story = "\n\n".join([self.story, "\n".join(actions)])
        while True:
            try:
                if n == 1:
                    raise NotImplementedError
                    raw_resp = call_gpt(
                        prompt=AutoToM_prompts.propose_1.format(story=story),
                        llm_name=self.llm_name,
                    )
                    hypos = raw_resp.replace("```json", "").replace("```", "").strip()
                    hypos = eval(hypos)
                    assert isinstance(hypos, dict)
                    assert hypos.keys() == {"objects", "target", "p"}
                else:
                    raw_resp = call_gpt(
                        prompt="\n\n".join[
                            AutoToM_prompts.question,
                            AutoToM_prompts.propose_n.format(story=story, n=n),
                        ],
                        llm_name=self.llm_name,
                    )
                    hypos = raw_resp.replace("```json", "").replace("```", "").strip()
                    hypos = eval(hypos)
                    assert isinstance(hypos, list)
                    assert all(isinstance(h, dict) for h in hypos)
                    assert all(h.keys() == {"objects", "target", "p"} for h in hypos)
                    hypos = hypos[:n]
                    assert len(hypos) == n
                    hypos_str_key = dict()
                    partition = sum(h["p"] for h in hypos)
                    for h in hypos:
                        p = h.pop("p")
                        assert isinstance(p, float)
                        hypos_str_key[str(h)] = p / partition
                    return hypos_str_key
            except (AssertionError, SyntaxError) as e:
                self.saver.error(f"{e}: {raw_resp}")

    def get_prob_dist_autotom(self, actions):
        # TODO: check this
        # ^ 0. get num_particles self.goal_hypos
        assert len(actions) != 0
        if len(self.goal_hypos) == 0:
            self.goal_hypos = self.propose_goal(
                actions,
                n=self.num_particles,
            )
        else:
            # ^ 1. filter out low-confidence hypos
            self.goal_hypos = {
                h: p for h, p in self.goal_hypos.items() if p >= self.filter_thres
            }
            # ^ 2. propose new hypos if needed
            if len(self.goal_hypos) < self.num_particles:
                new_hypos = self.propose_goal(
                    actions,
                    n=self.num_particles,
                )
                for new_hypo, p in new_hypos.items():
                    if new_hypo not in self.goal_hypos:
                        self.goal_hypos[new_hypo] = p
                    if len(self.goal_hypos) == self.num_particles:
                        break
        # ^ 3. reweight hypos and normalize
        # if len(actions) != 1:
        if True:
            # story = self.story + "\n" + actions[-1]
            while True:
                try:
                    # self.update_state(actions[-1])
                    probs = self.call_autotom_simple(actions[-1])
                    # probs = call_autotom(
                    #     story,
                    #     AutoToM_prompts,
                    #     list(self.goal_hypos.keys()),
                    #     self.llm_name,
                    # )
                    l_probs, l_choices = len(probs), len(self.goal_hypos)
                    assert l_probs == l_choices, f"{l_probs = } | {l_choices = }"
                    # * normalize
                    partition = sum(probs)
                    if partition == 0:
                        probs = [1 / l_choices for _ in probs]
                    else:
                        probs = [p / partition for p in probs]
                    break
                except Exception as e:
                    self.saver.error(f"[call_autotom] {e}")
                    traceback.print_exc()
            self.saver.debug(f"{self.goal_hypos.keys()}")
            self.saver.debug(f"curr_p = {probs}")
            self.saver.debug(f"prev_p = {self.goal_hypos.values()}")
            partition = sum(
                p * p_old for p, p_old in zip(probs, self.goal_hypos.values())
            )
            self.goal_hypos = {
                h: p * p_old / partition
                for p, (h, p_old) in zip(probs, self.goal_hypos.items())
            }
            self.saver.debug(f"combined_p = {self.goal_hypos.values()}")
            self.goal_hypos = {
                h: p
                for i, (h, p) in enumerate(self.goal_hypos.items())
                # * remove the whole particle if current prob[i] is low
                if probs[i] >= self.filter_thres
            }
            self.saver.debug(f"filtered_p = {self.goal_hypos.values()}")
        partition = sum(self.goal_hypos.values())
        self.goal_hypos = {h: p / partition for h, p in self.goal_hypos.items()}
        return self.goal_hypos

    def call_autotom_simple(self, current_action):
        choices = list(self.goal_hypos.keys())
        probs = []

        for choice in choices:
            # estimate the action likelihood
            prompt2 = f"""\
Based on the current environment state and the person's overall goal, what is the likelihood that the person would take the action described above?

Some hints: An action is highly likely if it directly contributes to the overall goal—for example, walking toward a goal object, grabbing a goal object, or putting it in/on the intended location.

If the action involves grabbing an object not mentioned in the goal, or placing an object somewhere other than the target location specified in the goal, the likelihood should be 0.

Please respond with a float number in [0, 1] inclusively, with no additional explanation.

Current environment state: {self.state}

Person's overall goal: {choice}

Action: {current_action}

Likelihood:"""
            raw_resp = call_gpt(prompt2, self.llm_name)
            likelihood = eval(raw_resp)
            probs.append(likelihood)
        return probs
