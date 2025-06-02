from virtualhome.simulation.evolving_graph.environment import Property

from agents.AutoToM import AutoToM
from agents.MCTS_agent import MCTS_agent
from utils import utils_environment as utils_env
from utils.utils_graph import EG


class AutoToM_agent(MCTS_agent):
    def __init__(self, *args, **kwargs):
        autotom_args = kwargs.pop("autotom_args")
        self.start_at_put = autotom_args.pop("start_at_put")
        self.autotom = AutoToM(**autotom_args)
        super().__init__(*args, **kwargs)
        self.agent_type = "AutoToM"

    def reset(self, *args, human_goal, human_init_room, **kwargs):
        super().reset(*args, **kwargs)
        # self.goal_hypos = dict()

        # self.predef_goals = {
        #     0: ("holds", "mug", self.agent_id, 1),
        #     1: ("on", "mug", 231, 1),
        #     2: ("holds", "spoon", self.agent_id, 1),
        #     3: ("on", "spoon", 231, 1),
        # }
        # self.predef_index = 0

        self.human_goal = human_goal
        self.human_init_room = human_init_room

        self.curr_goal = None
        self.curr_stage = "grab"  # "put"
        self.curr_hand = None
        self.autotom.saver = self.saver

    @staticmethod
    def goal_to_str(goal):
        if goal is None:
            return None
        else:
            # {"holds_mug_2": 1} -> "holds_mug_2"
            goal_name, x = list(goal.items())[0]
            if isinstance(x, int):
                return str({goal_name: x})
            else:
                return str({goal_name: x["count"]})

    @staticmethod
    def goal_to_obj_class(goal):
        if goal is None:
            return None
        else:
            # {"holds_mug_2": 1} -> "mug"
            return list(goal.keys())[0].split("_")[1]

    @staticmethod
    def goal_to_spec(goal, curr_gt_graph):
        if goal is None:
            return None
        else:
            return utils_env.convert_goal(goal, curr_gt_graph)

    def done_tgt_to_pred_and_id(self, eg):
        assert self.autotom.done_tgt is not None
        for predicate, tgt_id in TGT_LIST:
            tgt_node = eg[tgt_id]
            if tgt_node.class_name == self.autotom.done_tgt:
                return predicate, tgt_id
        raise ValueError(f"Invalid done_tgt: {self.autotom.done_tgt}")

    def check_goal_finished(self, goal, eg):
        assert len(goal) != 0
        if isinstance(list(goal.values())[0], int):
            goal = utils_env.convert_goal(goal, eg._dictionary)
        _, unsatisfied = utils_env.check_progress2(eg._dictionary, goal)
        finished = all(v <= 0 for v in unsatisfied.values())
        return finished

    def update_curr_goal(self, grab_goal, put_goal, eg):
        """
        TODO:
        this is ugly rule-based, can we make it recursive??
        """
        if self.curr_goal is not None:
            # ^ reset if achieved
            if self.check_goal_finished(self.curr_goal, eg):
                match self.curr_stage:
                    case "grab":
                        self.curr_stage = "put"
                        if put_goal is not None:
                            assert self.curr_hand == self.goal_to_obj_class(put_goal)
                        self.curr_goal = put_goal
                    case "put":
                        self.curr_stage = "grab"
                        self.curr_hand = self.goal_to_obj_class(grab_goal)
                        self.curr_goal = grab_goal
                    case _:
                        raise ValueError(f"Invalid stage: {self.curr_stage}")
            # else:
            #     # ^ if in_hand obj is already achieved
            #     match self.curr_stage:
            #         case "grab": # TODO: handle: grab wrong when 1 in obj_probs
            #             if self.autotom.done_tgt is not None:
            #                 predicate, tgt_id = self.done_tgt_to_pred_and_id(eg)
            #                 intend_goal = {
            #                     f"{predicate}_{self.curr_hand}_{tgt_id}": self.autotom.done_cnt
            #                 }
            #                 if self.check_goal_finished(intend_goal, eg):
            #                     self.curr_goal = None
            #                     self.curr_stage = "grab"
            #                     self.curr_hand = None
            #         case "put":
            #             ...
            #         case _:
            #             raise ValueError(f"Invalid stage: {self.curr_stage}")
        else:
            # ^ try to update if none
            match self.curr_stage:
                case "grab":
                    assert self.curr_stage == "grab"
                    self.curr_hand = self.goal_to_obj_class(grab_goal)
                    self.curr_goal = grab_goal
                case "put":
                    assert self.curr_stage == "put"
                    if put_goal is not None:
                        assert self.curr_hand == self.goal_to_obj_class(put_goal)
                    self.curr_goal = put_goal
                case _:
                    raise ValueError(f"Invalid stage: {self.curr_stage}")

        # ! rule override: if observed `done_tgt`
        # if (
        #     self.autotom.done_tgt is not None
        #     and self.curr_goal is not None
        #     and self.curr_stage == "put"
        # ):
        #     predicate, tgt_id = self.done_tgt_to_pred_and_id(eg)
        #     self.curr_goal = {
        #         f"{predicate}_{self.curr_hand}_{tgt_id}": self.autotom.done_cnt
        #     }

        self.curr_goal = self.goal_to_spec(self.curr_goal, eg._dictionary)

    def do_nothing(self):
        # dict_keys(['plan', 'subgoals', 'belief', 'belief_room', 'obs'])
        return None, dict(plan=[None])

    def get_action(self):
        curr_gt_graph = self.saver.episode_saved_info["graph"][-1]
        human_actions = self.saver.episode_saved_info["action"][0]
        eg = EG(curr_gt_graph)
        if self.start_at_put and any("[put" in a for a in human_actions):
            # * prepare `self.autotom.story` and `actions`
            actions = self.prepare_autotom(eg, human_actions)
            # * get candidate `obj` and `tgt`
            obj, tgt = self.autotom.step(actions)
            # * compose candidate `grab_goal` and `put_goal`
            grab_goal, put_goal = self.compose_goal(obj, tgt, eg)
            # * update `self.curr_goal`
            self.update_curr_goal(grab_goal, put_goal, eg)

            self.saver.debug(
                f"[Helper.plan] {self.goal_to_str(self.curr_goal)} | {self.curr_stage} | {self.curr_hand}"
            )

        self.saver.debug(f"[Hand] human={eg[1].holds()}, helper={eg[2].holds()}")

        if self.curr_goal is None:
            return self.do_nothing()
        else:
            return super().get_action(curr_gt_graph, self.curr_goal)

        # self.lg.debug(f"[Helper] inferred_goal = {human_goal}")
        # _, unsatisfied = utils_env.check_progress2(curr_gt_graph, gt_human_goal)
        # self.lg.debug(f"[GT] {unsatisfied = }")

        # human_goal = utils_env.convert_goal(human_goal, curr_gt_graph)
        # return super().get_action(curr_gt_graph, self.curr_goal, *args, **kwargs)
        # return super().get_action(obs, self.curr_goal, *args, **kwargs)

    def compose_goal(self, obj, tgt, eg):
        """
        Note: `obj` and `tgt` are both `class_name`
        """
        if obj is None:
            grab_goal = None
        else:
            grab_goal = {f"holds_{obj}_{self.agent_id}": 1}

        if tgt is None or self.curr_hand is None:
            # * not confident / grab nothing
            put_goal = None
        else:
            # * get `predicate` and `tgt_id`
            for predicate, tgt_id in TGT_LIST:
                tgt_node = eg[tgt_id]
                if tgt_node.class_name == tgt:
                    break
            else:
                tgt_node = list(eg.get_nodes_by_attr("class_name", tgt))[0]
                tgt_id = tgt_node.id
                if Property.CONTAINERS in tgt_node.properties:
                    predicate = "inside"
                elif Property.SURFACES in tgt_node.properties:
                    predicate = "on"
                else:
                    if tgt_node.category == "Rooms":
                        predicate = "inside"
                    else:
                        raise ValueError(f"Invalid target node: {tgt_node}")

            put_goal = {f"{predicate}_{self.curr_hand}_{tgt_id}": self.autotom.done_cnt}

        return grab_goal, put_goal

    def prepare_autotom(self, eg, human_actions):
        """
        prepare `self.autotom.story`, `self.autotom.done_cnt`, `actions`
        """
        # * 1. prepare `self.autotom.story`
        if not hasattr(self.autotom, "story"):
            story_belief = list(self.belief.edge_belief.values())
            story_ctnr_ids = story_belief[0]["INSIDE"][0][1:]
            story_srfc_ids = story_belief[0]["ON"][0][1:]

            story = eg.story(story_ctnr_ids, story_srfc_ids)
            self.autotom.story = story
        story_human = eg.agent_state_natlang()
        self.autotom.state = self.autotom.story + "\n\n" + story_human

        # * 2. prepare `self.autotom.done_cnt`
        if not hasattr(self.autotom, "done_cnt"):
            cnts = [v["count"] for v in self.human_goal.values()]
            assert all(cnts[0] == c for c in cnts), cnts
            self.autotom.done_cnt = cnts[0]

        # * 3. prepare `actions`
        actions = eg.actions_to_natlang(human_actions, self.human_init_room)
        return actions
