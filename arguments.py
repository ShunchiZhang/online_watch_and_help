import argparse


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
        default=640,
        help="The width of the image to save",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=480,
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
        "--num_tries",
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

    args = parser.parse_args()
    return args
