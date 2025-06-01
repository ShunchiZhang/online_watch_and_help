import atexit


# @ray.remote
class ArenaMP(object):
    def __init__(self, env, agents, saver):
        self.env = env
        self.agents = agents
        self.saver = saver
        atexit.register(self.env.close)

    def reset(self, task_id=None, helper_use_gt_goal=None):
        ob = None
        while ob is None:
            ob = self.env.reset(task_id=task_id, helper_use_gt_goal=helper_use_gt_goal)

        for it, agent in enumerate(self.agents):
            match agent.agent_type:
                case "MCTS":
                    agent.reset(self.env.full_graph, seed=agent.seed)
                case "AutoToM":
                    raise NotImplementedError("AutoToM is not implemented")
        return ob

    def get_actions(self, obs):
        actions = dict()
        agents_info = dict()

        for it, agent in enumerate(self.agents):
            goal_spec = self.env.get_goal2(
                self.env.task_goal[it],
                self.env.agent_goals[it],
            )

            match agent.agent_type:
                case "MCTS":
                    actions[it], agents_info[it] = agent.get_action(
                        obs[it],
                        goal_spec,
                        opponent_subgoal=None,
                        length_plan=5,
                        must_replan=False,
                    )
                case "AutoToM":
                    raise NotImplementedError("AutoToM is not implemented")

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
        self.saver.episode_saved_info = {
            "task_id": self.env.task_id,
            "env_id": self.env.env_id,
            "task_name": self.env.task_name,
            "gt_goals": self.env.task_goal[0],
            "goals": self.env.task_goal,
            "init_unity_graph": self.env.init_graph,
            "goals_finished": [],
            "obs": {i: [] for i in range(len(self.agents))},
            "graph": [self.env.init_unity_graph],
            "action": {i: [] for i in range(len(self.agents))},
            "plan": {i: [] for i in range(len(self.agents))},
            "belief": {i: [] for i in range(len(self.agents))},
            "belief_room": {i: [] for i in range(len(self.agents))},
            "belief_graph": {i: [] for i in range(len(self.agents))},
        }

        self.save_camera_img(self.env.steps)
        while True:
            (obs, reward, done, infos), actions, agents_info = self.step()
            self.save_camera_img(self.env.steps)

            for ith_agent in range(len(self.agents)):
                plan = agents_info[ith_agent]["plan"]
                self.saver.info(f"[{self.env.steps}] agent {ith_agent}: {plan}")
            self.saver.flush()

            self.saver.record_step(infos, actions, agents_info)

            if done:
                break

        self.saver.episode_saved_info["success"] = infos["finished"]
        self.saver.episode_saved_info["steps"] = self.env.steps
