from collections import Counter

from rich.pretty import pretty_repr

from agents.AutoToM import AutoToM
from agents.MCTS_agent import MCTS_agent
from utils import utils_environment as utils_env
from utils.utils_graph import EG, ENV_ID_TO_TARGET_NAME_TO_PREP, item, parse_string


class AutoToM_agent(MCTS_agent):
    def __init__(self, *args, **kwargs):
        autotom_args = kwargs.pop("autotom_args")
        self.grab_thres = autotom_args.pop("grab_thres")
        self.put_thres = autotom_args.pop("put_thres")
        self.start_at_put = autotom_args.pop("start_at_put")
        self.autotom = AutoToM(**autotom_args)

        super().__init__(*args, **kwargs)

        self.agent_type = "AutoToM"

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.autotom.saver = self.saver
        self.curr_goal = None

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
            parsed = parse_string(action)
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

        if not self.start_at_put or len(human_done) > 0:
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
                # ^ 1. get particles
                state, actions = self.prepare_autotom(curr_gt_graph, human_actions)
                particles = self.autotom.step(state, actions)
                self.saver.debug(f"[overall]\n{pretty_repr(particles)}")

                # ^ 2. minus done and ongoing particles
                particles.minus_objects(human_done + human_grab)
                particles.minus_objects(helper_done + helper_grab)
                self.saver.debug(f"[remain]\n{pretty_repr(particles)}")

                # ^ 3. update self.curr_goal
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
                env_id = self.task_meta["env_id"]
                put = probs.most_common(1)[0][0]
                verb, put_id = ENV_ID_TO_TARGET_NAME_TO_PREP[env_id][put]
                grab, _ = item(helper_grab)
                self.curr_goal = {f"{verb}_{grab}_{put_id}": 1}

    def prepare_autotom(self, curr_gt_graph, human_actions):
        """
        prepare `state` and `actions`
        """
        eg = EG(curr_gt_graph)

        # ^ 1. prepare `state`
        if not hasattr(self.autotom, "story"):
            story_belief = list(self.belief.edge_belief.values())
            story_ctnr_ids = story_belief[0]["INSIDE"][0][1:]
            story_srfc_ids = story_belief[0]["ON"][0][1:]

            story = eg.story(
                story_ctnr_ids,
                story_srfc_ids,
                self.task_meta["task_name"],
                self.task_meta["env_id"],
            )
            self.autotom.story = story
        # * state: human close to ..., holds ...
        state = eg.agent_state_natlang(agent_id=1, name="Human")

        # ^ 2. prepare `actions`
        human_init_room = self.saver.episode_saved_info["init_rooms"][0]
        actions = eg.actions_to_natlang(human_actions, human_init_room)
        return state, actions
