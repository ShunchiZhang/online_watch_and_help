import atexit


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
            "hands": [],
            "executed": [],
            "llm_time": 0,
            "llm_dollar": 0,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
        }

    def reset(self, episode_id, helper_goal_type, seed):
        ob = None
        while ob is None:
            ob = self.env.reset(
                task_id=episode_id,
                helper_goal_type=helper_goal_type,
                seed=seed,
            )
        self.reset_saver()

        for it, agent in enumerate(self.agents):
            agent.seed = seed + it
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

        executed = self.saver.episode_saved_info["executed"]
        must_replan = len(executed) > 1 and any(
            (a is not None) and (("grab" in a) or ("put" in a))
            for a in executed[-1][1].values()
        )

        for it, agent in enumerate(self.agents):
            match agent.agent_type:
                case "MCTS":
                    actions[it], agents_info[it] = agent.get_action(
                        obs=obs[it],
                        # * `self.env.goal_spec` has additional 'final' 'reward' keys
                        # * than `self.env.task_goal`
                        goal_spec=self.env.goal_spec[it],
                        must_replan=must_replan,
                    )
                case "AutoToM":
                    # * AutoToM_agent will collect info from `saver`
                    actions[it], agents_info[it] = agent.get_action(
                        obs=obs[it],
                        must_replan=must_replan,
                    )

        return actions, agents_info

    def step(self):
        obs = self.env.get_observations()
        actions, agents_info = self.get_actions(obs)
        obs, reward, done, env_info = self.env.step(actions)
        return (obs, reward, done, env_info), actions, agents_info

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
            (obs, reward, done, env_info), actions, agents_info = self.step()
            self.save_camera_img(self.env.steps)

            steps = self.env.steps
            graph = self.env.get_graph()
            self.saver.record_step(steps, env_info, actions, agents_info, graph)

            if done:
                break

        success = env_info["finished"]
        self.saver.episode_saved_info["success"] = success
        self.saver.episode_saved_info["steps"] = self.env.steps

        pbar.remove_task(pbar_step)

        return success
