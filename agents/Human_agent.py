from utils.utils_graph import (
    check_progress,
)


class Human_agent:
    def __init__(self, *args, **kwargs):
        self.agent_type = "Human"

    def reset(self, gt_graph):
        self.init_gt_graph = gt_graph

    def get_action(self, obs):
        graphs = self.saver.episode_saved_info["graph"]

        prev_actions = self.saver.episode_saved_info["action"]
        human_actions = prev_actions[0]
        helper_actions = prev_actions[1]

        human_done, human_grab, human_touched = check_progress(human_actions)
        helper_done, helper_grab, helper_touched = check_progress(helper_actions)

        return action
