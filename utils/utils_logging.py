import json
import logging
import os
import pickle
from pathlib import Path

import cv2
from rich.logging import RichHandler
from virtualhome.simulation.environment.unity_environment import (
    UnityEnvironment as BaseUnityEnvironment,
)


def prettier(path):
    cmd = [
        f"cd {path.parent}",
        f"prettier --write {path.name}",
        "cd -",
    ]
    os.system(" && ".join(cmd))
    return cmd


def format_row(obj_list, spec, sep=""):
    formats = [format(str(obj), spec) for obj in obj_list]
    return sep.join(formats)


class LevelBasedFormatter(logging.Formatter):
    def __init__(self, formats, default_format="%(message)s"):
        super().__init__()
        self.formats = formats
        self.default_format = default_format

    def format(self, record):
        fmt = self.formats.get(record.levelno, self.default_format)
        formatter = logging.Formatter(fmt)
        return formatter.format(record)


RES_PY_FORMATS = {
    logging.DEBUG: "# %(message)s",
    logging.INFO: "%(message)s",
    logging.WARNING: "# ^ %(message)s",
    logging.ERROR: "# ! %(message)s",
}


def get_my_logger(
    name="shunchi",
    level=logging.DEBUG,
    file_level=logging.DEBUG,
    file_format="%(asctime)s|%(levelname)s|%(message)s",
    file_name=None,
    rich_level=logging.DEBUG,
    rich_args=dict(
        show_level=True,
        show_path=True,
        show_time=False,
    ),
):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if len(logger.handlers) == 0:
        # * rich handler
        if "res_py" not in name:
            rich_handler = RichHandler(**rich_args)
            rich_handler.setLevel(rich_level)
            logger.addHandler(rich_handler)

        # * file handler
        if file_name is not None:
            file_handler = logging.FileHandler(file_name)
            file_handler.setLevel(file_level)
            if "res_py" not in name:
                file_handler.setFormatter(logging.Formatter(file_format))
            else:
                file_handler.setFormatter(LevelBasedFormatter(RES_PY_FORMATS))
            logger.addHandler(file_handler)

    return logger


class Saver:
    def __init__(self, name, record_dir, save_img):
        self.record_dir = record_dir
        self.log_filename = record_dir / f"{name}.log"
        self.logger = get_my_logger(name=name, file_name=self.log_filename)
        self._inherit_logging()

        self.info(f"cwd: {Path.cwd()}")
        self.info(f"logging to '{self.log_filename}'")

        self.img_w = save_img["image_width"]
        self.img_h = save_img["image_height"]
        camera_views = save_img["camera_views"]
        view_options = set(BaseUnityEnvironment.camera_mapping.keys())
        if camera_views.lower() == "all":
            self.camera_views = view_options
        elif camera_views in ("", None):
            self.camera_views = []
        else:
            if isinstance(camera_views, str):
                self.camera_views = [camera_views]
            else:
                self.camera_views = camera_views
        assert all(view in view_options for view in self.camera_views)

    def _inherit_logging(self):
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical
        self.log = self.logger.log
        self.exception = self.logger.exception

    def reset_run(self, run_id):
        self.run_id = run_id
        self.run_dir = self.record_dir / f"run_{run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.run_path = self.run_dir / "results.json"
        self.run_result = dict()  # episode_id: (success, steps)

    def reset_episode(self, episode_id):
        self.episode_id = episode_id
        self.episode_dir = self.run_dir / f"episode_{episode_id:02d}"
        self.episode_path = self.episode_dir / "result.json"
        self.episode_graph_path = self.episode_dir / "graph.pik"

        if self.episode_path.exists():
            with self.episode_path.open("r") as f:
                episode_result = json.load(f)
                self.record_episode(episode_result)
        else:
            self.info(f">>>>> start: {self.current_episode} >>>>>")

    def save_episode(self, episode_result):
        graph_keys = {
            "init_unity_graph",
            "belief",
            "belief_room",
            "belief_graph",
            "graph",
            "obs",
        }

        graph_data = {k: episode_result[k] for k in graph_keys if k in episode_result}
        with self.episode_graph_path.open("wb") as f:
            pickle.dump(graph_data, f)

        other_data = {k: v for k, v in episode_result.items() if k not in graph_keys}
        with self.episode_path.open("w") as f:
            json.dump(other_data, f, ensure_ascii=False)
        prettier(self.episode_path)
        self.record_episode(episode_result)

    def record_episode(self, episode_result):
        success = episode_result["success"]
        steps = episode_result["steps"]
        self.run_result[self.episode_id] = (success, steps)
        self.info(f"[{self.current_episode}] success: {success}, steps: {steps}")

    def save_run(self):
        failure_list = []
        steps_list = []
        for episode_id, (success, steps) in self.run_result.items():
            if success:
                steps_list.append(steps)
            else:
                failure_list.append(episode_id)

        avg_steps = sum(steps_list) / len(steps_list)
        num_failures = len(failure_list)
        total_episodes = len(self.run_result)
        self.info(f">>>>> summary: run_{self.run_id} >>>>>")
        self.info(f"avg_steps: {avg_steps:.1f}")
        self.info(f"failures: {failure_list} ({num_failures}/{total_episodes})")
        self.info(f"<<<<< summary: run_{self.run_id} <<<<<")

        with self.run_path.open("w") as f:
            json.dump(
                dict(
                    details=self.run_result,
                    avg_steps=avg_steps,
                    failures=failure_list,
                ),
                f,
                ensure_ascii=False,
            )
        prettier(self.run_path)

    def save_camera_img(self, img, agent_id, view, step):
        img_path = self.episode_dir / f"agent{agent_id}" / view / f"{step:04d}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(img_path, img)

    @property
    def current_episode(self):
        return f"run_{self.run_id}/episode_{self.episode_id:02d}"
