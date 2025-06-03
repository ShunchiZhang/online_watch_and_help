import pickle
from pathlib import Path

from agents.AutoToM_agent import AutoToM_agent
from agents.MCTS_agent import MCTS_agent
from arguments import get_args
from envs.arena_mp2 import ArenaMP
from envs.unity_environment import UnityEnvironment
from utils import utils_graph, utils_logging


class Runner:
    def __init__(self, args):
        self.args = args
        self._get_env_task_set()
        self._get_saver()
        self._get_agents()
        self._get_env()
        self.arena = ArenaMP(self.env, self.agents, self.saver)

    def _get_saver(self):
        self.args.record_dir = Path(self.args.record_dir) / self.args.dataset_path.stem

        self.args.record_dir.mkdir(parents=True, exist_ok=True)
        self.saver = utils_logging.Saver(
            logger_name=self.args.logger_name,
            record_dir=self.args.record_dir,
            save_img=dict(
                camera_views=self.args.save_camera_views,
                image_width=self.args.image_width,
                image_height=self.args.image_height,
            ),
            save_belief=False,
        )

    def _get_env_task_set(self):
        self.args.dataset_path = Path(self.args.dataset_path)
        with self.args.dataset_path.open("rb") as f:
            env_task_set = pickle.load(f)
        env_task_set = utils_graph.fix_graph(env_task_set)
        env_task_set = utils_graph.fix_multiple_location(
            env_task_set, verbose=False, drop_env=True
        )
        self.env_task_set = env_task_set

    def _get_agents(self):
        args_agent_common = dict(
            recursive=False,
            max_episode_length=20,  # MCTS:expand()
            num_simulation=20,
            max_rollout_steps=5,
            c_init=0.1,
            c_base=100,
            num_samples=1,
            logging=True,
            logging_graphs=True,
            agent_params=dict(
                open_cost=0,
                should_close=False,
                walk_cost=0.05,
                belief=dict(
                    forget_rate=0,
                    belief_type="uniform",
                ),
            ),
        )
        args_agents = []

        for i in range(self.args.num_agents):
            args_agent = dict(agent_id=i + 1, char_index=i, **args_agent_common)
            args_agent["agent_params"]["obs_type"] = self.args.obs_type[i]

            if self.args.debug:
                args_agent["num_particles"] = (
                    1 if self.args.obs_type[i] == "full" else 3
                )
                args_agent["num_processes"] = 0
            else:
                num_particles = (
                    1 if self.args.obs_type[i] == "full" else self.args.num_particles
                )
                args_agent["num_particles"] = num_particles
                args_agent["num_processes"] = num_particles

            args_agents.append(args_agent)

        match self.args.helper_class.lower():
            case "autotom":
                agent_classes = [MCTS_agent, AutoToM_agent]
                args_agents[1]["autotom_args"] = dict(
                    grab_thres=self.args.autotom_thres_grab,
                    put_thres=self.args.autotom_thres_put,
                    filter_thres=self.args.autotom_thres_filter,
                    num_particles=self.args.autotom_num_particles,
                    llm_name=self.args.autotom_llm_name,
                    method=self.args.autotom_method,
                    start_at_put=self.args.autotom_start_at_put,
                )
            case _:
                agent_classes = [MCTS_agent, MCTS_agent]

        self.agents = [
            agent_class(**args_agent)
            for agent_class, args_agent in zip(agent_classes, args_agents)
        ]

    def _get_env(self):
        self.env = UnityEnvironment(
            num_agents=len(self.agents),
            max_episode_length=self.args.max_steps,
            port_id=0,
            convert_goal=True,
            env_task_set=self.env_task_set,
            observation_types=self.args.obs_type,
            use_editor=self.args.use_editor,
            executable_args=dict(
                file_name=self.args.executable_file,
                x_display="0",
                no_graphics=False,
            ),
            base_port=self.args.base_port,
        )

    def run(self):
        if self.args.episode_ids is None:
            self.args.episode_ids = list(range(len(self.env_task_set)))
        if self.args.debug_len is not None:
            self.args.episode_ids = self.args.episode_ids[: self.args.debug_len]

        episode_ids = self.args.episode_ids

        with self.saver.pbar as pbar:
            pbar_run = pbar.add_task("run", total=self.args.num_runs)

            for ith_run in range(self.args.num_runs):
                pbar.update(pbar_run, advance=1)
                pbar_episode = pbar.add_task("episode", total=len(episode_ids))
                self.saver.reset_run(ith_run)

                for episode_id in episode_ids:
                    pbar.update(pbar_episode, advance=1)

                    self.saver.reset_episode(episode_id, self.env_task_set[episode_id])
                    if self.saver.episode_path.exists():
                        continue

                    self.arena.reset(episode_id, ith_run, helper_use_gt_goal=False)
                    self.arena.run()
                    self.saver.save_episode()

                self.saver.save_run()
                pbar.remove_task(pbar_episode)


if __name__ == "__main__":
    runner = Runner(args=get_args())
    runner.run()
