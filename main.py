import pickle
from pathlib import Path

from agents.GnP_agent import GnP_agent
from agents.Human_agent import Human_agent
from agents.MCTS_agent import MCTS_agent
from arguments import get_args
from envs.arena import Arena
from envs.unity_environment import UnityEnvironment
from utils.utils_exception import check_unity_error, handle
from utils.utils_graph import fix_graph, fix_multiple_location
from utils.utils_logging import Saver


class Runner:
    def __init__(self, args):
        self.args = args
        self._get_env_task_set()
        self._get_saver()
        self._get_agents()
        self._get_env()
        self.arena = Arena(self.env, self.agents, self.saver)

    def _get_saver(self):
        if self.args.num_agents == 1:
            method = "single"
        else:
            if self.args.helper_class == "MCTS":
                if self.args.helper_goal_type == "unknown":
                    raise ValueError("MCTS helper cannot infer goals")
                elif self.args.helper_goal_type == "gt":
                    method_suffix = "oracle_goal"
                elif self.args.helper_goal_type == "random":
                    method_suffix = "random_goal"
                else:
                    raise ValueError(f"{self.args.helper_goal_type = }")

            elif self.args.helper_class == "GnP":
                proposer_name = self.args.autotom_proposer_name.split("/")[-1]
                if self.args.autotom_method == "autotom":
                    estimator_name = self.args.autotom_estimator_name.split("/")[-1]
                    method_suffix = f"autotom_Q={proposer_name}_P={estimator_name}"
                elif self.args.autotom_method == "llm":
                    method_suffix = proposer_name
                else:
                    raise ValueError(f"{self.args.autotom_method = }")

            elif self.args.helper_class == "Human":
                method_suffix = "human"

            else:
                raise ValueError(f"{self.args.helper_class = }")

            method = f"{self.args.helper_class}_{method_suffix}"

        self.args.record_dir = (
            Path(self.args.record_dir) / self.args.dataset_path.stem / method
        )
        self.args.record_dir.mkdir(parents=True, exist_ok=True)

        self.saver = Saver(
            logger_name=self.args.logger_name,
            record_dir=self.args.record_dir,
            save_img=dict(
                camera_views=self.args.save_camera_views,
                image_width=self.args.image_width,
                image_height=self.args.image_height,
            ),
            save_belief=False,
            process_id=self.args.process_id,
        )

    def _get_env_task_set(self):
        self.args.dataset_path = Path(self.args.dataset_path)
        with self.args.dataset_path.open("rb") as f:
            env_task_set = pickle.load(f)
        env_task_set = fix_graph(env_task_set)
        env_task_set = fix_multiple_location(env_task_set, verbose=False, drop_env=True)
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

        self.agents = []

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

            if i == 0 or self.args.helper_class == "MCTS":
                self.agents.append(MCTS_agent(**args_agent))
            else:
                match self.args.helper_class:
                    case "GnP":
                        args_agent["autotom_args"] = dict(
                            filter_thres=self.args.autotom_thres_filter,
                            num_particles=self.args.autotom_num_particles,
                            proposer_name=self.args.autotom_proposer_name,
                            estimator_name=self.args.autotom_estimator_name,
                            method=self.args.autotom_method,
                            hide_helper_history=self.args.autotom_hide_helper_history,
                            disable_estimation=self.args.autotom_disable_estimation,
                        )
                        args_agent["agent_args"] = dict(
                            thres_grab=self.args.gnp_thres_grab,
                            thres_put=self.args.gnp_thres_put,
                            start_at_put=self.args.gnp_start_at_put,
                        )
                        self.agents.append(GnP_agent(**args_agent))
                    case "Human":
                        self.agents.append(Human_agent(**args_agent))
                    case _:
                        raise ValueError(f"Invalid config: {self.args.helper_class}")

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
                x_display=self.args.display,
                no_graphics=False,
                timeout_wait=30,
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
                self.saver.reset_run(ith_run)
                if self.saver.run_path.exists():
                    pbar.update(pbar_run, advance=1)
                    continue

                pbar_episode = pbar.add_task("episode", total=len(episode_ids))
                for episode_id in episode_ids:
                    self.saver.reset_episode(episode_id, self.env_task_set[episode_id])

                    if not self.saver.episode_path.exists():
                        for ith_retry in range(self.args.num_retries):
                            if ith_retry != 0:
                                msg = f"retry {ith_retry}: {self.saver.current_episode}"
                                self.saver.warning(msg)

                            try:
                                self.arena.reset(
                                    episode_id=episode_id,
                                    helper_goal_type=self.args.helper_goal_type,
                                    seed=len(self.agents) * ith_run * ith_retry,
                                )
                                success = self.arena.run()
                                if success:
                                    break

                            except Exception as e:
                                e = check_unity_error(e)
                                handle(e, self.saver, exc_info=True)
                                self.saver.remove_pbar_task("step")

                    self.saver.save_episode()
                    pbar.update(pbar_episode, advance=1)

                self.saver.save_run()
                pbar.update(pbar_run, advance=1)
                pbar.remove_task(pbar_episode)


if __name__ == "__main__":
    runner = Runner(args=get_args())
    runner.run()
