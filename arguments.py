import argparse


def add_autotom_args(parser):
    parser.add_argument(
        "--helper_class",
        type=str,
        default="AutoToM",
        choices=["AutoToM", "MCTS"],
        help="The class of the helper to use",
    )
    parser.add_argument(
        "--autotom_thres_grab",
        type=float,
        default=0.80,
        help="The threshold for the grab action in AutoToM",
    )
    parser.add_argument(
        "--autotom_thres_put",
        type=float,
        default=0.60,
        help="The threshold for the put action in AutoToM",
    )
    parser.add_argument(
        "--autotom_thres_filter",
        type=float,
        default=0.20,
        help="The threshold for the particle filter in AutoToM",
    )
    parser.add_argument(
        "--autotom_num_particles",
        type=int,
        default=5,
        help="The number of particles to use in AutoToM",
    )
    parser.add_argument(
        "--autotom_llm_name",
        type=str,
        default="gpt-4o",
        choices=["gpt-4o", "o3-mini", "gpt-4o-mini"],
        help="The name of the LLM to use in AutoToM",
    )
    parser.add_argument(
        "--autotom_method",
        type=str,
        choices=["autotom", "llm"],
        default="autotom",
        help="The method to use in AutoToM",
    )
    parser.add_argument(
        "--autotom_start_at_put",
        action="store_true",
        default=False,
        help="Whether to start AutoToM at the beginning or wait for human's first putback action",
    )
    return parser


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--record_dir",
        type=str,
        default="logs",
        help="The directory to save the results",
    )
    parser.add_argument(
        "--save_camera_views",
        type=str,
        nargs="+",
        default="third_behind",
        choices=[
            "third_front",
            "third_isometric",
            "first_front",
            "third_behind",
            "third_oblique",
            "first_right",
            "first_left",
            "first_back",
        ],
        help="The camera views to save",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=160,  # 640,
        help="The width of the image to save",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=120,  # 480,
        help="The height of the image to save",
    )
    parser.add_argument(
        "--logger_name",
        type=str,
        default="main",
        help="The name of the logger. Logging to logger_name.log",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=2,
        help="The number of agents to test",
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=3,
        help="The number of particles to use",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Whether to run in debug mode (#processes=0)",
    )
    parser.add_argument(
        "--debug_len",
        type=int,
        default=None,
        help="The number of episodes to run in debug mode",
    )
    parser.add_argument(
        "--episode_ids",
        type=int,
        nargs="+",
        default=None,
        help="The episode ids to run",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="The number of times to run the same environment to reduce variance",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="The path of the environments where we test",
    )
    parser.add_argument(
        "--obs_type",
        type=str,
        nargs="+",
        default=["full", "full"],
        choices=["full", "rgb", "visibleid", "partial"],
        help="Observation types to use. Can specify multiple types.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="number of steps",
    )
    parser.add_argument(
        "--executable_file",
        type=str,
        default="../executable/linux_exec_v3.x86_64",
    )
    parser.add_argument(
        "--base_port",
        type=int,
        default=8080,
    )
    parser.add_argument(
        "--display",
        type=str,
        default="2",
    )
    parser.add_argument(
        "--use_editor",
        action="store_true",
        default=False,
        help="whether to use an editor or executable",
    )

    parser = add_autotom_args(parser)

    args = parser.parse_args()
    return args
