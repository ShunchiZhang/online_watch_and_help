import atexit

from utils.utils_graph import EG


# @ray.remote
class ArenaMP(object):
    def __init__(self, env, agents, saver):
        self.env = env
        self.agents = agents
        self.saver = saver
        atexit.register(self.env.close)

    def reset_saver(self):
        self.saver.episode_saved_info = {
            "task_id": self.env.task_id,
            "env_id": self.env.env_id,
            "task_name": self.env.task_name,
            "gt_goals": self.env.task_goal[0],
            "goals": self.env.task_goal,
            "init_rooms": {i: self.env.init_rooms[i] for i in range(len(self.agents))},
            "init_unity_graph": self.env.init_graph,
            "goals_finished": [],
            "obs": {i: [] for i in range(len(self.agents))},
            "graph": [self.env.init_unity_graph],
            "action": {i: [] for i in range(len(self.agents))},
            "plan": {i: [] for i in range(len(self.agents))},
            "belief": {i: [] for i in range(len(self.agents))},
            "belief_room": {i: [] for i in range(len(self.agents))},
            "belief_graph": {i: [] for i in range(len(self.agents))},
            "llm_time": 0,
            "llm_dollar": 0,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
        }

    def reset(self, task_id, ith_run, helper_use_gt_goal):
        ob = None
        while ob is None:
            ob = self.env.reset(task_id=task_id, helper_use_gt_goal=helper_use_gt_goal)
        self.reset_saver()

        for it, agent in enumerate(self.agents):
            agent.seed = (it + ith_run * 2) * 5
            agent.saver = self.saver
            match agent.agent_type:
                case "MCTS":
                    agent.reset(self.env.full_graph)
                case "AutoToM":
                    agent.reset(self.env.full_graph)

        return ob

    def get_actions(self, obs):
        actions = dict()
        agents_info = dict()

        for it, agent in enumerate(self.agents):
            match agent.agent_type:
                case "MCTS":
                    actions[it], agents_info[it] = agent.get_action(
                        obs=obs[it],
                        # * `self.env.goal_spec` has additional 'final' 'reward' keys
                        # * than `self.env.task_goal`
                        goal_spec=self.env.goal_spec[it],
                    )
                case "AutoToM":
                    # * AutoToM_agent will collect info from `saver`
                    actions[it], agents_info[it] = agent.get_action(obs=obs[it])

        return actions, agents_info

    def step(self):
        obs = self.env.get_observations()
        actions, agents_info = self.get_actions(obs)
        obs, reward, done, infos = self.env.step(actions)
        return (obs, reward, done, infos), actions, agents_info

    def save_camera_img(self, step):
        for agent_id in range(len(self.agents)):
            for view in self.saver.camera_views:
                obs = self.env.get_observation(
                    agent_id=agent_id,
                    obs_type="image",
                    info=dict(
                        view=view,
                        image_width=self.saver.img_w,
                        image_height=self.saver.img_h,
                    ),
                )
                self.saver.save_camera_img(obs, agent_id, view, step)

    def run(self):
        self.save_camera_img(self.env.steps)

        pbar = self.saver.pbar
        pbar_step = pbar.add_task("step", total=self.env.max_episode_length)

        while True:
            pbar.update(pbar_step, advance=1)
            (obs, reward, done, infos), actions, agents_info = self.step()
            self.save_camera_img(self.env.steps)

            eg = EG(self.env.get_graph())
            for ith_agent in range(len(self.agents)):
                plan = agents_info[ith_agent]["plan"]
                in_hand = eg[ith_agent + 1].holds()
                self.saver.info(f"[{self.env.steps}]agent{ith_agent}/{in_hand}/{plan}")
            self.saver.flush()

            self.saver.record_step(infos, actions, agents_info)

            if done:
                break

        self.saver.episode_saved_info["success"] = infos["finished"]
        self.saver.episode_saved_info["steps"] = self.env.steps

        pbar.remove_task(pbar_step)
