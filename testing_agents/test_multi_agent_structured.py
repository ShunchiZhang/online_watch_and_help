import pickle
from pathlib import Path

from rich.pretty import pretty_repr
from tqdm import tqdm

from agents.MCTS_agent_particle_v2_instance import MCTS_agent_particle_v2_instance
from algos.arena_mp2 import ArenaMP
from arguments import get_args
from envs.unity_environment import UnityEnvironment
from utils import utils_environment as utils_env
from utils import utils_graph, utils_logging

if __name__ == "__main__":
    args = get_args()

    args.dataset_path = Path(args.dataset_path)
    env_task_set = pickle.load(args.dataset_path.open("rb"))
    env_task_set = utils_graph.fix_graph(env_task_set)
    env_task_set = utils_graph.fix_multiple_location(
        env_task_set, verbose=False, drop_env=True
    )

    args_agent_common = dict(
        recursive=False,
        max_episode_length=20,  # MCTS_particles_v2_instance:expand()
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
    for i in range(args.num_agents):
        args_agent = dict(agent_id=i + 1, char_index=i, **args_agent_common)
        args_agent["agent_params"]["obs_type"] = args.obs_type[i]

        if args.debug:
            args_agent["num_particles"] = 1 if args.obs_type[i] == "full" else 3
            args_agent["num_processes"] = 0
        else:
            num_particles = 1 if args.obs_type[i] == "full" else args.num_particles
            args_agent["num_particles"] = num_particles
            args_agent["num_processes"] = num_particles

        args_agents.append(args_agent)

    agents = [
        lambda x, y: MCTS_agent_particle_v2_instance(**args_agent)
        for args_agent in args_agents
    ]

    def env_fn(arena_id):
        return UnityEnvironment(
            num_agents=len(agents),
            max_episode_length=args.max_steps,
            port_id=arena_id,
            convert_goal=True,
            env_task_set=env_task_set,
            observation_types=args.obs_type,
            use_editor=args.use_editor,
            executable_args=dict(
                file_name=args.executable_file,
                x_display="0",
                no_graphics=False,
            ),
            base_port=args.base_port,
        )

    arena = ArenaMP(
        max_number_steps=args.max_steps,
        arena_id=0,
        environment_fn=env_fn,
        agent_fn=agents,
        use_sim_agent=False,
        save_belief=False,
    )

    args.record_dir = Path(args.record_dir) / args.dataset_path.stem
    args.record_dir.mkdir(parents=True, exist_ok=True)
    saver = utils_logging.Saver(
        name=args.logger_name,
        record_dir=args.record_dir,
        save_img=dict(
            camera_views=args.save_camera_views,
            image_width=args.image_width,
            image_height=args.image_height,
        ),
    )

    episode_ids = list(range(len(env_task_set)))

    for ith_try in range(args.num_tries):
        # ^ run
        saver.reset_run(ith_try)

        for episode_id in tqdm(episode_ids):
            # ^ episode
            saver.reset_episode(episode_id)
            if saver.episode_path.exists():
                continue

            for it_agent, agent in enumerate(arena.agents):
                agent.seed = (it_agent + ith_try * 2) * 5

            arena.reset(episode_id)
            env_task = env_task_set[episode_id]
            human_goal = utils_env.convert_goal(
                env_task["task_goal"][0], env_task["init_graph"]
            )
            saver.info(f"Human Goal: {pretty_repr(human_goal, indent_size=2)}")
            episode_result = arena.run(saver=saver)
            saver.save_episode(episode_result)

        saver.save_run()
