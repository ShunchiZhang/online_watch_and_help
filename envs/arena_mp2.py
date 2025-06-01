import atexit
import copy
import random
import traceback

import numpy as np


# @ray.remote
class ArenaMP(object):
    def __init__(
        self,
        max_number_steps,
        arena_id,
        environment_fn,
        agent_fn,
        use_sim_agent=False,
        save_belief=True,
        saver=None,
    ):
        self.agents = []
        self.sim_agents = []
        self.save_belief = save_belief
        self.env_fn = environment_fn
        self.agent_fn = agent_fn
        self.arena_id = arena_id
        self.num_agents = len(agent_fn)
        self.task_goal = None
        self.use_sim_agent = use_sim_agent
        print("Init Env")
        self.env = environment_fn(arena_id)
        assert self.env.num_agents == len(
            agent_fn
        ), "The number of agents defined and the ones in the env defined mismatch"
        for agent_type_fn in agent_fn:
            self.agents.append(agent_type_fn(arena_id, self.env))
            if self.use_sim_agent:
                self.sim_agents.append(agent_type_fn(arena_id, self.env))

        self.max_episode_length = self.env.max_episode_length
        self.max_number_steps = max_number_steps
        self.saved_info = None
        atexit.register(self.close)

        self.saver = saver

    def close(self):
        print(traceback.print_exc())

        print(traceback.print_stack())

        self.env.close()

    def reset(self, task_id=None, helper_use_gt_goal=None):
        ob = None
        while ob is None:
            ob = self.env.reset(task_id=task_id, helper_use_gt_goal=helper_use_gt_goal)
        print(ob.keys(), self.num_agents)

        for it, agent in enumerate(self.agents):
            if "MCTS" in agent.agent_type or "Random" in agent.agent_type:
                agent.reset(
                    ob[it], self.env.full_graph, self.env.task_goal, seed=agent.seed
                )
                if self.use_sim_agent:
                    self.sim_agents[it].reset(
                        ob[it],
                        self.env.full_graph,
                        self.env.task_goal,
                        seed=self.agents[1 - it].seed,
                    )
            else:
                agent.reset(self.env.full_graph)
                if self.use_sim_agent:
                    self.sim_agents.reset(self.env.full_graph)
        return ob

    def get_actions(self, obs):
        dict_actions, dict_info = {}, {}

        for it, agent in enumerate(self.agents):
            goal_spec = self.env.get_goal2(self.task_goal[it], self.env.agent_goals[it])

            match agent.agent_type:
                case "MCTS":
                    dict_actions[it], dict_info[it] = agent.get_action(
                        obs[it],
                        goal_spec,
                        opponent_subgoal=None,
                        length_plan=5,
                        must_replan=False,
                    )
                case "AutoToM":
                    raise NotImplementedError("AutoToM is not implemented")

        return dict_actions, dict_info

    def step(self):
        obs = self.env.get_observations()
        dict_actions, dict_info = self.get_actions(obs)
        step_info = self.env.step(dict_actions)
        return step_info, dict_actions, dict_info

    def save_camera_img(self, step):
        # * save images
        for agent_id in range(self.num_agents):
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

    def run(self, random_goal=False, pred_goal=None):
        """
        self.task_goal: goal inference
        self.env.task_goal: ground-truth goal
        """
        self.task_goal = copy.deepcopy(self.env.task_goal)
        if random_goal:
            for predicate in self.env.task_goal[0]:
                u = random.choice([0, 1, 2])
                self.task_goal[0][predicate] = u
                self.task_goal[1][predicate] = u
        if pred_goal is not None:
            self.task_goal = copy.deepcopy(pred_goal)

        saved_info = {
            "task_id": self.env.task_id,
            "env_id": self.env.env_id,
            "task_name": self.env.task_name,
            "gt_goals": self.env.task_goal[0],
            "goals": self.task_goal,
            "action": {0: [], 1: []},
            "plan": {0: [], 1: []},
            "finished": None,
            "init_unity_graph": self.env.init_graph,
            "goals_finished": [],
            "belief": {0: [], 1: []},
            "belief_room": {0: [], 1: []},
            "belief_graph": {0: [], 1: []},
            "graph": [self.env.init_unity_graph],
            "obs": [],
        }

        success = False
        num_failed = 0
        num_repeated = 0
        prev_action = None
        self.saved_info = saved_info
        step = 0
        prev_agent_position = np.array([0, 0, 0]).astype(np.float32)
        self.save_camera_img(step)
        while True:
            step += 1
            (obs, reward, done, infos), actions, agent_info = self.step()
            self.save_camera_img(step)
            # ipdb.set_trace()
            new_agent_position = np.array(
                list(infos["graph"]["nodes"][0]["bounding_box"]["center"])
            ).astype(np.float32)
            distance = np.linalg.norm(new_agent_position - prev_agent_position)
            step_failed = infos["failed_exec"]
            if actions[0] == prev_action:
                num_repeated += 1
                if distance < 0.3:
                    num_nomove += 1
            else:
                prev_action = actions[0]
                num_repeated = 0
                num_nomove = 0
            if step_failed:
                num_failed += 1
            else:
                num_failed = 0
            if num_failed > 10 or num_repeated > 20:
                self.saver.error(f"Many failures: {num_failed}, {num_repeated}")
                done = True
                infos["finished"] = False

            self.saver.info(f"step: {step}")
            # self.saver.info(f"--> action: {actions}")
            for ith_agent in range(self.num_agents):
                plan = agent_info[ith_agent]["plan"]
                self.saver.info(f"--> agent {ith_agent}: {plan}")
            self.saver.flush()
            # print("Goals:", self.env.task_goal)
            # self.saver.info("Action: ", actions, new_agent_position)
            prev_agent_position = new_agent_position
            # logging.info(" | ".join(actions.values()))
            success = infos["finished"]
            if "satisfied_goals" in infos:
                saved_info["goals_finished"].append(infos["satisfied_goals"])
            for agent_id, action in actions.items():
                saved_info["action"][agent_id].append(action)

            if "graph" in infos:
                saved_info["graph"].append(infos["graph"])

            for agent_id, info in agent_info.items():
                # if 'belief_graph' in info:
                #    saved_info['belief_graph'][agent_id].append(info['belief_graph'])
                if self.save_belief:
                    if "belief_room" in info:
                        saved_info["belief_room"][agent_id].append(info["belief_room"])
                    if "belief" in info:
                        saved_info["belief"][agent_id].append(info["belief"])
                if "plan" in info:
                    saved_info["plan"][agent_id].append(info["plan"][:3])
                if "obs" in info:
                    # print("TEST", len(info['obs']), len(saved_info['graph'][-2]['nodes']))
                    saved_info["obs"].append([node["id"] for node in info["obs"]])
                    # print([node['states'] for node in info['obs'] if node['id'] == 103])
                    # ipdb.set_trace()
                # if len(saved_info['obs']) > 1 and set(saved_info['obs'][0]) != set(saved_info['obs'][1]):
                #    ipdb.set_trace()

            # ipdb.set_trace()
            if done:
                break
            self.saved_info = saved_info

        saved_info["obs"].append([node["id"] for node in obs[0]["nodes"]])
        # saved_info['obs'].append()

        saved_info["success"] = success
        saved_info["steps"] = self.env.steps
        self.saved_info = saved_info
        return saved_info
