import json
import logging
import os
import pickle
from pathlib import Path

import cv2
from rich.logging import RichHandler
from rich.pretty import pretty_repr
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from virtualhome.simulation.environment.unity_environment import (
    UnityEnvironment as BaseUnityEnvironment,
)


def get_by_agent_id(data, agent_id):
    if int(agent_id) in data:
        return data[int(agent_id)]
    elif str(agent_id) in data:
        return data[str(agent_id)]
    else:
        raise ValueError(f"Agent ID {agent_id} not found in data")


def prettier(path):
    cmd = [
        f"cd {path.parent}",
        f"prettier --write {path.name}",
        "cd -",
    ]
    os.system(" && ".join(cmd) + " > /dev/null 2>&1")
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
            file_handler.setFormatter(logging.Formatter(file_format))
            logger.addHandler(file_handler)

            file_handler_info = logging.FileHandler(file_name.with_suffix(".info.log"))
            file_handler_info.setLevel(logging.INFO)
            file_handler_info.setFormatter(logging.Formatter(file_format))
            logger.addHandler(file_handler_info)

            file_handler_warn = logging.FileHandler(file_name.with_suffix(".warn.log"))
            file_handler_warn.setLevel(logging.WARNING)
            file_handler_warn.setFormatter(logging.Formatter(file_format))
            logger.addHandler(file_handler_warn)

    return logger


class Saver:
    def __init__(self, logger_name, record_dir, save_img, save_belief, process_id):
        self.record_dir = record_dir
        if process_id is not None:
            logger_name = f"{logger_name}_{process_id:02d}"
        self._init_logging(logger_name)

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

        self.save_belief = save_belief

    def flush(self):
        for handler in self.logger.handlers:
            handler.flush()
            try:
                os.fsync(handler.stream.fileno())
            except Exception:
                pass

    def _init_logging(self, name):
        log_filename = self.record_dir / f"{name}.log"
        self.logger = get_my_logger(name=name, file_name=log_filename)

        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical
        self.log = self.logger.log
        self.exception = self.logger.exception

        self.info(f"cwd: {Path.cwd()}")
        self.info(f"logging to '{log_filename}'")

        self.pbar = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            "<",
            TimeRemainingColumn(),
        )

    def reset_run(self, run_id):
        self.run_id = run_id
        self.run_dir = self.record_dir / f"run_{run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.run_path = self.run_dir / "results.json"
        self.run_result = dict()  # episode_id: (success, steps)

    def reset_episode(self, episode_id, env_task):
        self.episode_id = episode_id
        self.episode_dir = self.run_dir / f"episode_{episode_id:02d}"
        self.episode_path = self.episode_dir / "result.json"
        self.episode_graph_path = self.episode_dir / "graph.pik"
        self.episode_eval_path = self.episode_dir / "eval.json"

        if self.episode_path.exists():
            with self.episode_path.open("r") as f:
                other_data = json.load(f)
            with self.episode_graph_path.open("rb") as f:
                graph_data = pickle.load(f)
            with self.episode_eval_path.open("r") as f:
                eval_data = json.load(f)
            self.episode_saved_info = {**other_data, **graph_data, **eval_data}

        else:
            self.info(f">>>>> start: {self.current_episode} >>>>>")
            # ! prevent circular import
            from utils.utils_graph import EG, Goal

            env_id = env_task["env_id"]
            task_name = env_task["task_name"]
            goal = Goal(
                env_task["task_goal"][0],
                EG(env_task["init_graph"]),
            )

            self.info(f"[{task_name}] (apt_id={env_id})\n{goal}")

    def save_episode(self):
        saved_info = self.episode_saved_info
        graph_keys = {
            "init_unity_graph",
            "belief",
            "belief_room",
            "belief_graph",
            "graph",
            "obs",
        }
        eval_keys = {
            "success",
            "steps",
            "num_goals",
            "helper_total_grab",
            "helper_total_put",
            "helper_correct_grab",
            "helper_correct_put",
            "helper_valid_help",
            "helper_valid_step",
            "llm_time",
            "llm_dollar",
            "llm_input_tokens",
            "llm_output_tokens",
        }

        graph_data = {k: saved_info[k] for k in graph_keys if k in saved_info}
        with self.episode_graph_path.open("wb") as f:
            pickle.dump(graph_data, f)

        self.save_episode_eval()
        eval_data = {k: saved_info[k] for k in eval_keys if k in saved_info}
        with self.episode_eval_path.open("w") as f:
            json.dump(eval_data, f, ensure_ascii=False)
        prettier(self.episode_eval_path)

        other_data = {
            k: v for k, v in saved_info.items() if k not in (graph_keys | eval_keys)
        }
        with self.episode_path.open("w") as f:
            json.dump(other_data, f, ensure_ascii=False)
        prettier(self.episode_path)

    def save_episode_eval(self):
        from utils.utils_graph import item, parse_action

        metrics = dict()
        action_seq = []
        metrics["helper_total_grab"] = 0
        metrics["helper_total_put"] = 0
        metrics["helper_correct_grab"] = 0
        metrics["helper_correct_put"] = 0
        metrics["helper_valid_step"] = self.episode_saved_info["steps"]

        gt_goals = self.episode_saved_info["gt_goals"]
        gt_grab = {k.split("_")[1] for k in gt_goals}  # class_name
        gt_put = item({k.split("_")[2] for k in gt_goals})  # id
        num_goals = sum([spec["count"] for spec in gt_goals.values()])
        metrics["num_goals"] = num_goals
        metrics["helper_valid_help"] = num_goals

        # for t, executed in enumerate(self.episode_saved_info["executed"]):
        #     _, executed, _, _ = executed

        #     executed_action = executed.values()
        #     if len(executed_action) == 1:
        #         a_h, a_r = item(executed_action), None
        #     elif len(executed_action) == 2:
        #         a_h, a_r = executed_action
        #     else:
        #         raise ValueError(f"{len(executed_action)}-agent is not supported")

        for t, a_list in enumerate(zip(*self.episode_saved_info["action"].values())):
            a_h = a_list[0]
            parsed_h = parse_action(a_h)
            match parsed_h[0]:
                case "grab":
                    action_seq.append(f"[{t + 1:2d}] human {a_h}")
                case "putin" | "putback":
                    action_seq.append(f"[{t + 1:2d}] human {a_h}")
                    metrics["helper_valid_help"] -= 1

            if len(a_list) == 1:
                continue

            a_r = a_list[1]
            parsed_r = parse_action(a_r)
            match parsed_r[0]:
                case "grab":
                    action_seq.append(f"[{t + 1:2d}] helper {a_r}")

                    metrics["helper_total_grab"] += 1
                    if parsed_r[1] in gt_grab:
                        metrics["helper_correct_grab"] += 1
                        action_seq[-1] += " (correct: grab)"
                case "putin" | "putback":
                    action_seq.append(f"[{t + 1:2d}] helper {a_r}")
                    metrics["helper_total_put"] += 1
                    if parsed_r[4] == gt_put:
                        metrics["helper_correct_put"] += 1
                        if parsed_r[1] in gt_grab:
                            action_seq[-1] += " (correct: grab & put)"
                        else:
                            action_seq[-1] += " (correct: put)"
                case None:
                    metrics["helper_valid_step"] -= 1

        # * as helper won't grab human-touched objects, it's human planner bug
        metrics["helper_valid_help"] = max(0, metrics["helper_valid_help"])

        self.episode_saved_info["action_seq"] = action_seq

        self.episode_saved_info.update(metrics)
        all_metrics = dict(
            metrics,
            success=self.episode_saved_info["success"],
            steps=self.episode_saved_info["steps"],
            llm_time=self.episode_saved_info.get("llm_time", 0),
            llm_dollar=self.episode_saved_info.get("llm_dollar", 0),
            llm_input_tokens=self.episode_saved_info.get("llm_input_tokens", 0),
            llm_output_tokens=self.episode_saved_info.get("llm_output_tokens", 0),
        )
        self.run_result[self.episode_id] = all_metrics
        self.info(f"[{self.current_episode}.metrics]\n{pretty_repr(all_metrics)}")

    def record_step(self, steps, env_info, actions, agent_info, graph):
        # ! prevent circular import
        from utils.utils_graph import EG, dedup_list, subgoal_string_to_tuple

        eg = EG(graph)

        saved_info = self.episode_saved_info

        # ^ agent info
        for agent_id, info in agent_info.items():
            # * save
            action = actions[agent_id]
            saved_info["action"][agent_id].append(action)
            if self.save_belief:
                if "belief_graph" in info:
                    saved_info["belief_graph"][agent_id].append(info["belief_graph"])
                if "belief_room" in info:
                    saved_info["belief_room"][agent_id].append(info["belief_room"])
                if "belief" in info:
                    saved_info["belief"][agent_id].append(info["belief"])
            if "plan" in info:
                saved_info["plan"][agent_id].append(info["plan"])
            if "subgoals" in info:
                subgoals = dedup_list(sum(info["subgoals"], []))
                subgoals = [subgoal_string_to_tuple(s) for s in subgoals]
                saved_info["subgoals"][agent_id].append(subgoals)
            if "obs" in info:
                saved_info["obs"][agent_id].append([node["id"] for node in info["obs"]])

        hands = dict()
        for agent_id in range(len(actions)):
            hands_objects = [(n.id, n.class_name) for n in eg[agent_id + 1].holds()]
            assert len(hands_objects) <= 1
            hands[agent_id] = hands_objects
        saved_info["hands"].append(hands)

        curr_subgoals = [s[-1] for s in saved_info["subgoals"].values()]
        self.warning(f"[{steps:2d}] {'-' * 80}")
        self.warning(f" action  {format_row(actions.values(), '<50')}")
        self.warning(f"  goal   {format_row(curr_subgoals, '<50')}")
        self.warning(f"  hand   {format_row(hands.values(), '<50')}")

        # ^ env info
        if "satisfied_goals" in env_info:
            saved_info["goals_finished"].append(env_info["satisfied_goals"])
        if "graph" in env_info:
            saved_info["graph"].append(env_info["graph"])

        executed_ids = env_info["executed_script"].keys() - env_info["message"].keys()
        executed_ids = sorted(list(executed_ids))
        executed_actions = {
            k: v if k in executed_ids else None for k, v in actions.items()
        }

        nonempty_actions = {k: v for k, v in actions.items() if v is not None}
        failed_ids = nonempty_actions.keys() - set(executed_ids)
        failed_actions = {k: v for k, v in nonempty_actions.items() if k in failed_ids}

        saved_info["executed"].append(
            [
                actions,
                executed_actions,
                failed_actions,
                env_info["message"],  # env_info['failed_exec']
            ]
        )

        if len(executed_ids) != len(nonempty_actions):
            self.error(f"[{steps}] {failed_actions = }")
            if len(env_info["message"]) > 0:
                self.error(f"[{steps}] {env_info['message'] = }")

        self.episode_saved_info = saved_info
        return saved_info

    def record_cost(self, cost, name):
        if name is not None:
            self.debug(f"[{name}] {cost}")
        self.episode_saved_info["cost"] += cost

    def save_run(self):
        failure_list = []
        steps_list = []
        num_goals = 0
        valid_help = 0
        llm_stats = dict(time=0, dollar=0, input_tokens=0, output_tokens=0)
        for episode_id, metrics in self.run_result.items():
            llm_stats["time"] += metrics["llm_time"] / 3600
            llm_stats["dollar"] += metrics["llm_dollar"]
            llm_stats["input_tokens"] += metrics["llm_input_tokens"]
            llm_stats["output_tokens"] += metrics["llm_output_tokens"]
            if metrics["success"]:
                steps_list.append(metrics["steps"])
                num_goals += metrics["num_goals"]
                valid_help += metrics["helper_valid_help"]
            else:
                failure_list.append(episode_id)

        avg_steps = sum(steps_list) / len(steps_list) if len(steps_list) > 0 else -1
        valid_help_rate = valid_help / num_goals if num_goals > 0 else -1
        num_failures = len(failure_list)
        total_episodes = len(self.run_result)
        self.info(f">>>>> summary: run_{self.run_id} >>>>>")
        self.info(f"avg_steps: {avg_steps:.1f}")
        self.info(f"total_valid_help: {valid_help}/{num_goals} ({valid_help_rate:.2f})")
        self.info(f"failures: {failure_list} ({num_failures}/{total_episodes})")
        self.info(f"llm_stats: {llm_stats}")
        self.info(f"<<<<< summary: run_{self.run_id} <<<<<")

        detail_keys = [
            "success",
            "steps",
            "num_goals",
            "helper_valid_help",
            "helper_total_put",
        ]
        with self.run_path.open("w") as f:
            json.dump(
                dict(
                    avg_steps=avg_steps,
                    failures=failure_list,
                    llm=llm_stats,
                    details_keys=detail_keys,
                    details={
                        episode_id: [result[k] for k in detail_keys]
                        for episode_id, result in self.run_result.items()
                    },
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
