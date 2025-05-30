import argparse


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument('--dataset_path', type=str, help="The path of the environments where we test")
    parser.add_argument(
        '--obs_type',
        type=str,
        default='partial',
        choices=['full', 'rgb', 'visibleid', 'partial'],
    )
    parser.add_argument(
        '--max-episode-length',
        type=int,
        default=200,
        help='number of episodes')
    parser.add_argument(
        '--executable_file', type=str,
        default='../executable/linux_exec_v3.x86_64')
    parser.add_argument(
        '--base-port', type=int, default=8080)
    parser.add_argument(
        '--display', type=str, default="2")

    parser.add_argument('--use-editor', action='store_true', default=False,
                        help='whether to use an editor or executable')
    args = parser.parse_args()
    return args
