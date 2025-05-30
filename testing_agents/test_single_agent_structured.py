import pickle
import json
import numpy as np
from pathlib import Path
from envs.unity_environment import UnityEnvironment
from agents.MCTS_agent_particle_v2_instance import MCTS_agent_particle_v2_instance
from arguments import get_args
from algos.arena_mp2 import ArenaMP
from utils import utils_environment as utils_env

if __name__ == '__main__':
    args = get_args()

    save_data = True
    num_tries = 1

    args.dataset_path = Path(args.dataset_path)
    env_task_set = pickle.load(args.dataset_path.open('rb'))

    for env in env_task_set:
        init_gr = env['init_graph']
        gbg_can = [node['id'] for node in init_gr['nodes'] if node['class_name'] in ['garbagecan', 'clothespile']]
        init_gr['nodes'] = [node for node in init_gr['nodes'] if node['id'] not in gbg_can]
        init_gr['edges'] = [edge for edge in init_gr['edges'] if edge['from_id'] not in gbg_can and edge['to_id'] not in gbg_can]
        for node in init_gr['nodes']:
            if node['class_name'] == 'cutleryfork':
                node['obj_transform']['position'][1] += 0.1

    args.record_dir = Path() / "exp" / args.dataset_path.stem
    args.record_dir.mkdir(parents=True, exist_ok=True)
    args.obs_type = 'full'

    executable_args = dict(
        file_name=args.executable_file,
        x_display="0",
        no_graphics=True
    )

    id_run = 0
    episode_ids = list(range(len(env_task_set)))

    S = [[] for _ in range(len(episode_ids))]
    L = [[] for _ in range(len(episode_ids))]
    
    test_results = {}

    args_common = dict(
        recursive=False,
        max_episode_length=20,
        num_simulation=20,
        max_rollout_steps=5,
        c_init=0.1,
        c_base=100,
        num_samples=1,
        num_processes=0, 
        num_particles=1 if args.obs_type == 'full' else 3,
        logging=True,
        logging_graphs=True,
        agent_params=dict(
            obs_type=args.obs_type,
            open_cost=0,
            should_close=False,
            walk_cost=0.05,
            belief=dict(
                forget_rate=0,
                belief_type="uniform",
            )
        )
    )

    args_agent1 = dict(agent_id=1, char_index=0, **args_common)
    args_agent2 = dict(agent_id=2, char_index=1, **args_common)

    agents = [
        lambda x, y: MCTS_agent_particle_v2_instance(**args_agent1),
        lambda x, y: MCTS_agent_particle_v2_instance(**args_agent2),
    ]

    def env_fn(env_id):
        return UnityEnvironment(
            num_agents=len(agents),
            max_episode_length=args.max_episode_length,
            port_id=env_id,
            convert_goal=True,
            env_task_set=env_task_set,
            observation_types=[args.obs_type, args.obs_type],
            use_editor=args.use_editor,
            executable_args=executable_args,
            base_port=args.base_port
        )

    arena = ArenaMP(args.max_episode_length, id_run, env_fn, agents, save_belief=False)

    for ith_try in range(num_tries):

        cnt = 0
        steps_list, failed_tasks = [], []

        if (args.record_dir / 'results.pik').exists():
            test_results = pickle.load(open(args.record_dir / 'results.pik', 'rb'))
        else:
            test_results = {}

        for episode_id in episode_ids:

            log_file_name = args.record_dir / 'e{}r{}.pik'.format(episode_id, ith_try)
            if log_file_name.exists():
                continue

            print('episode:', episode_id)

            for it_agent, agent in enumerate(arena.agents):
                agent.seed = (it_agent + ith_try * 2) * 5

            arena.reset(episode_id)
            env_task = env_task_set[episode_id]
            agent_goal = utils_env.convert_goal(env_task['task_goal'][0], env_task['init_graph'])
            print("Agent Goal", agent_goal)
            success, steps, saved_info = arena.run()

            print('-------------------------------------')
            print('success' if success else 'failure')
            print('steps:', steps)
            print('-------------------------------------')
            if not success:
                failed_tasks.append(episode_id)
            else:
                steps_list.append(steps)
            is_finished = 1 if success else 0

            if len(saved_info['obs']) > 0 and save_data:
                pickle.dump(saved_info, open(log_file_name, 'wb'))
            else:
                if save_data:
                    with open(log_file_name, 'w+') as f:
                        f.write(json.dumps(saved_info, indent=4))

            S[episode_id].append(is_finished)
            L[episode_id].append(steps)
            test_results[episode_id] = {'S': S[episode_id],
                                        'L': L[episode_id]}
        
        print('average steps (finishing the tasks):', np.array(steps_list).mean() if len(steps_list) > 0 else None)
        print('failed_tasks:', failed_tasks)
        if save_data:
            pickle.dump(test_results, (args.record_dir / 'results.pik').open('wb'))
