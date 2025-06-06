import copy
from collections import Counter

from rich.pretty import pretty_repr

from agents import AutoToM_prompts as prompts
from agents.AutoToM import AutoToM
from agents.MCTS_agent import MCTS_agent
from utils import utils_environment as utils_env
from utils.utils_graph import (
    ENV_ID_TO_TARGET_NAME_TO_ID,
    TARGET_NAME_TO_PREP,
    item,
    parse_action,
)


class AutoToM_agent(MCTS_agent):
    def __init__(self, *args, **kwargs):
        autotom_args = kwargs.pop("autotom_args")
        self.grab_thres = autotom_args.pop("grab_thres")
        self.put_thres = autotom_args.pop("put_thres")
        self.start_at_put = autotom_args.pop("start_at_put")
        self.autotom = AutoToM(**autotom_args)

        super().__init__(*args, **kwargs)

        self.agent_type = "AutoToM"

    def reset(self, gt_graph):
        super().reset(gt_graph)

        self.curr_goal = None
        self.goal_particles = prompts.GoalParticles(particles=[])

        self.autotom.saver = self.saver
        self.autotom.reset(gt_graph, self.belief)

    @property
    def curr_verb(self):
        return None if self.curr_goal is None else item(self.curr_goal)[0].split("_")[0]

    def check_progress(self, actions):
        """
        Input:  history actions
        Output: done counter (put), ongoing counter (grab)
        """
        done_counter = Counter()
        grab_counter = Counter()
        touched_obj_ids = set()
        for action in actions:
            if action is None:
                continue
            parsed = parse_action(action)
            match parsed[0]:
                case "grab":
                    obj = parsed[1]
                    grab_counter[obj] += 1

                    touched_obj_ids.add(int(parsed[2]))

                case "putin" | "putback":
                    obj = parsed[1]
                    done_counter[obj] += 1

                    grab_counter[obj] -= 1
                    if grab_counter[obj] == 0:
                        grab_counter.pop(obj)
        return done_counter, grab_counter, touched_obj_ids

    def get_action(self, obs):
        curr_gt_graph = self.saver.episode_saved_info["graph"][-1]

        prev_actions = self.saver.episode_saved_info["action"]
        human_actions = prev_actions[0]
        helper_actions = prev_actions[1]

        human_done, human_grab, human_touched = self.check_progress(human_actions)
        helper_done, helper_grab, helper_touched = self.check_progress(helper_actions)

        start_autotom = False
        if self.start_at_put and len(human_done) > 0:
            start_autotom = True
        elif not self.start_at_put and len(human_actions) > 0:
            start_autotom = True

        if start_autotom:
            # ^ 1. maintain particles for every step
            keep_particles = (
                len(human_actions) >= 2 and human_actions[-1] == human_actions[-2]
            )
            if not keep_particles:
                self.goal_particles = self.autotom.step(
                    curr_gt_graph, human_actions, self.goal_particles
                )

            # ^ 2. decide to replan or not
            should_replan = dict(
                grab=(
                    (sum(helper_grab.values()) == 0)
                    and (
                        (self.curr_goal is None) or (self.curr_verb in {"inside", "on"})
                    )
                ),
                put=(
                    (sum(helper_grab.values()) == 1)
                    and ((self.curr_goal is None) or (self.curr_verb in {"holds"}))
                ),
            )
            if any(should_replan.values()):
                # ^ 3. minus done and ongoing particles
                particles = copy.deepcopy(self.goal_particles)
                particles.minus_objects(human_done + human_grab)
                particles.minus_objects(helper_done + helper_grab)
                if not keep_particles:
                    self.saver.debug(f"[remain]\n{pretty_repr(particles.to_natlang())}")

                # ^ 4. update self.curr_goal
                self.update_curr_goal(particles, helper_grab, should_replan)

        if self.curr_goal is None:
            # dict_keys(['plan', 'subgoals', 'belief', 'belief_room', 'obs'])
            return None, dict(plan=[None])
        else:
            goal_spec = utils_env.convert_goal(self.curr_goal, curr_gt_graph)

            match self.curr_verb:
                # * for grab, exlucde human touched objects by hacking goal_spec
                case "holds":
                    candidates = goal_spec[item(goal_spec.keys())]["grab_obj_ids"]
                    candidates = [c for c in candidates if c not in human_touched]
                    goal_spec[item(goal_spec.keys())]["grab_obj_ids"] = candidates

                # * for put, always assure the goal is not finished by human
                case "inside" | "on":
                    goal_spec = utils_env.convert_goal(self.curr_goal, curr_gt_graph)
                    satisfied, _ = utils_env.check_progress2(curr_gt_graph, goal_spec)
                    _, satisfied = item(satisfied)
                    self.curr_goal[item(self.curr_goal.keys())] = len(satisfied) + 1

            self.saver.debug(f"[helper.goal] {self.curr_goal}")
            return super().get_action(obs, goal_spec)

    def update_curr_goal(self, particles, helper_grab, should_replan):
        # ^ plan for holds {holds_???_2: 1}
        if should_replan["grab"]:
            # * get the most likely object
            probs = Counter()
            for particle in particles.particles:
                for object in particle.objects:
                    probs[object.type] += particle.p
            self.saver.debug(f"[hold.probs] {probs}")

            # * update goal
            if len(probs) == 0 or probs.most_common(1)[0][1] < self.grab_thres:
                self.curr_goal = None
            else:
                grab = probs.most_common(1)[0][0]
                self.curr_goal = {f"holds_{grab}_{self.agent_id}": 1}

        # ^ plan for put {on/inside_???_???: ???}
        if should_replan["put"]:
            # * get the most likely object
            probs = Counter()
            for particle in particles.particles:
                probs[particle.target.type] += particle.p
            self.saver.debug(f"[put.probs] {probs}")

            # * update goal
            if len(probs) == 0 or probs.most_common(1)[0][1] < self.put_thres:
                self.curr_goal = None
            else:
                env_id = self.saver.episode_saved_info["env_id"]
                put = probs.most_common(1)[0][0]
                put_id = ENV_ID_TO_TARGET_NAME_TO_ID[env_id][put]
                verb = TARGET_NAME_TO_PREP[put]
                grab, _ = item(helper_grab)
                self.curr_goal = {f"{verb}_{grab}_{put_id}": 1}

                if put_id is None:
                    env_id = self.saver.episode_saved_info["env_id"]
                    self.saver.error(f"{put} doesn't exist in apt {env_id}")
                    self.curr_goal = None
