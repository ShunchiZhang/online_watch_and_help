import copy
import itertools
import random
from collections import Counter
from datetime import datetime

from rich import print
from tqdm import tqdm

GEN_META = {
    0: dict(
        custom_setup_table=dict(
            TGT_LIST=[
                ["on", 231],  # (231)kitchentable   < (205)kitchen
                ["on", 371],  # (371)coffeetable    < (335)livingroom
            ],
            OBJ_LIST=[
                ["plate", "cutleryfork"],
                ["plate", "wineglass"],
                ["plate", "waterglass"],
                ["wineglass", "waterglass"],
            ],
            ROOM_LIST=["livingroom", "kitchen", "bedroom", "bathroom"],
            EXCLUDE_OBJS=dict(
                plate={314},
                wineglass={197, 198},
            ),
            EXCLUDE_ACTIONS={
                ("putback", "kitchencounter"),
            },
        )
    ),
    1: dict(),
    2: dict(),
    3: dict(),
    4: dict(),
    5: dict(),
    6: dict(),
    7: dict(),
}

"""
valid tasks:
{
    'setup_table': {('on', 231, 'kitchentable'), ('on', 372, 'coffeetable')},
    'put_fridge': {('inside', 247, 'fridge'), ('inside', 155, 'fridge')},
    'prepare_food': {('on', 136, 'kitchentable'), ('inside', 152, 'stove'), ('on', 194, 'kitchentable')},
    'put_dishwasher': {('inside', 154, 'dishwasher')}
}
{
    'setup_table': {'plate', 'wineglass', 'cutleryfork', 'waterglass'},
    'put_fridge': {'salmon', 'apple', 'cupcake', 'pudding'},
    'prepare_food': {'salmon', 'apple', 'cupcake', 'pudding'},
    'put_dishwasher': {'plate', 'wineglass', 'cutleryfork', 'waterglass'}
}
"""

ENV_ID, TASK_NAME = 0, "custom_setup_table"
TGT_LIST = GEN_META[ENV_ID][TASK_NAME]["TGT_LIST"]
OBJ_LIST = GEN_META[ENV_ID][TASK_NAME]["OBJ_LIST"]
ROOM_LIST = GEN_META[ENV_ID][TASK_NAME]["ROOM_LIST"]
OBJ_SET = {o for obj_names in OBJ_LIST for o in obj_names}


def argmax_dict_items(d):
    return max(d.items(), key=lambda x: x[1])


def post_init_args(args):
    if args.convert_tp:
        """
        read from `args.convert_tp` dir, calculate traj len
        """
        args.exp_id = f"cvtp_{args.convert_tp}"
        args.num_reruns = 1
    else:
        exp_id_parts = [f"c{args.num_obj_cnt}", f"r{args.num_reruns}"]

        if args.disable_tp:
            exp_id_parts.append("notp")
        else:
            exp_id_parts.append("tp")

        if args.dummy_helper:
            exp_id_parts.append("single")
        else:
            exp_id_parts.append(args.helper_method[:3])
            exp_id_parts.append(f"{args.helper_llm_model[:2]}")

        if args.exp_id == "":
            exp_id_parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
        else:
            exp_id_parts.append(args.exp_id)

        args.exp_id = "_".join(exp_id_parts)
    return args


def modify_goals_of_env_task_set(env_task_set, episode_ids):
    """
    used for
    - new_datasets/all.json
    - shunchi/dataset.pik
    - shunchi/dataset_v2.pik
    """
    filtered_ids = dict()
    for episode_id in tqdm(episode_ids):
        datum = env_task_set[episode_id]

        # * balancing task classes
        task_name = datum["task_name"].split("_and_")[0]
        if len(filtered_ids.get(task_name, [])) >= 3:
            continue

        if "0" in datum["task_goal"].keys():
            colab_goal = datum["task_goal"]["0"]
        elif 0 in datum["task_goal"].keys():
            colab_goal = datum["task_goal"][0]
        else:
            raise ValueError(datum["task_goal"].keys())

        # * all tgt_ids are the same
        tgt_ids = [g.split("_")[-1] for g in colab_goal.keys()]
        if not all(tgt_ids[0] == i for i in tgt_ids):
            continue

        # * 2 differnt objects, 1 cnt each
        if not len(colab_goal.keys()) >= 2:
            continue
        colab_goal = dict(list(colab_goal.items())[:2])
        colab_goal = {g: 1 for g, cnt in colab_goal.items()}

        # * remove donut-related tasks for mysterious reason
        if any("donut" in g for g in colab_goal.keys()):
            continue

        env_task_set[episode_id]["task_name"] = task_name
        env_task_set[episode_id]["task_goal"] = {0: colab_goal, 1: dict()}

        filtered_ids.setdefault(task_name, []).append(episode_id)

    filtered_ids = sum(list(filtered_ids.values()), [])
    env_task_set = [env_task_set[i] for i in filtered_ids]
    filtered_ids = list(range(len(env_task_set)))

    print("before filtering:", len(episode_ids))
    print(" after filtering:", len(filtered_ids))
    print(Counter([env_task_set[i]["task_name"] for i in filtered_ids]).most_common())
    return env_task_set, filtered_ids


def fixed_scenario_of_env_task_set(datum, enable_room_product):
    assert datum["env_id"] == ENV_ID
    env_task_set = []
    eg = EG(datum["init_graph"])
    task_name = TASK_NAME

    # * assure all objects are in the graph
    if not all(eg.has_class(o) for o in OBJ_SET):
        return [], []

    # * get valid goals (tgt_list X obj_list)
    valid_goals = []
    for (predicate, tgt_id), obj_names in itertools.product(TGT_LIST, OBJ_LIST):
        # * check if already (partially) satisfied
        sup_objs = eg[tgt_id].supports()
        sup_obj_names = {o.class_name for o in sup_objs}
        if any(o in sup_obj_names for o in obj_names):
            continue
        else:
            goal = {f"{predicate}_{o}_{tgt_id}": 1 for o in obj_names}
            valid_goals.append(goal)

    # * get valid rooms (2 out of 4)
    if enable_room_product:
        valid_rooms = itertools.combinations(ROOM_LIST, 2)
    else:
        valid_rooms = [datum["init_rooms"]]

    for room, goal in itertools.product(valid_rooms, valid_goals):
        # d["init_rooms"] = list(room)
        # d["task_goal"] = {0: goal, 1: dict()}
        new_datum = copy.deepcopy(datum)
        new_datum["task_name"] = task_name
        new_datum["task_goal"] = {0: goal, 1: dict()}
        new_datum["init_rooms"] = list(room)
        env_task_set.append(new_datum)

    filtered_ids = list(range(len(env_task_set)))
    return env_task_set, filtered_ids


def fixed_scenario_of_env_task_set__(datum):
    """
    used for
    - new_datasets/dataset_language_large.pik

    #nodes = 398, #edges = 633~637
    - 454: remotecontrol, potato, carrot, wineglass, ...
    - 455: toy, wine, beer, wineglass, remotecontrol, cellphone, book, potato, ...
    nodes (same across the dataset):
    - room:
        - bathroom:       11
        - bedroom:        73
        - kitchen:        205
        - livingroom:     335
    - surface:
        - coffeetable:    111, 371
        - kitchentable:   231
        - sofa:           368
        - desk:           108, 373
        - kitchencounter: 238
    - container:
        - fridge:         305
        - cabinet:        415
        - microwave:      313
        - stove:          311
        - kitchencabinet: 234, 235, 236, 237
    - objects:
        - mug:            194, 447
        - wineglass:      197, 198, 298, 299
        - waterglass:     64,  270, 274, 281, 282
        - plate:          61,  193, 199, 273, 277, 278, 285, 314
        - cutleryknife:   271, 275, 280, 283
        - cutleryfork:    272, 276, 279, 284
        - folder:         203, 204, 453


    CUSTOM TASK:
    cartesian product of:
    - target:
        - (231)kitchentable   < (205)kitchen
        - (238)kitchencounter < (205)kitchen
        - (371)coffeetable    < (335)livingroom
    - object:
        - mug, milk
        - mug, wineglass
        - wine, wineglass
    """
    task_name = "drink"
    predicate = "on"
    cnt = 1
    env_id = 0  # apartment id
    room_list = ["livingroom", "kitchen", "bedroom", "bathroom"]
    tgt_list = [231, 238, 371]
    obj_list = [
        "mug",
        "wineglass",
        "waterglass",
        "plate",
        "cutleryknife",
        "cutleryfork",
    ]

    # * get env_task_set[0]["init_graph"] & remove nodes & edges related to 454/455
    base_graph = datum["init_graph"]
    base_graph["nodes"] = [n for n in base_graph["nodes"] if n["id"] not in (454, 455)]
    base_graph["edges"] = [
        e
        for e in base_graph["edges"]
        if e["from_id"] not in (454, 455) and e["to_id"] not in (454, 455)
    ]

    dataset = dict()
    for episode_id in tqdm(episode_ids):
        env_task_set[episode_id]["task_name"] = task_name
        env_task_set[episode_id]["init_graph"] = base_graph

        # * create random set_table task
        rng = random.Random(episode_id)

        obj_names = rng.sample(obj_list, k=2)
        tgt_id = rng.choice(tgt_list)
        colab_goal = {f"{predicate}_{obj_name}_{tgt_id}": cnt for obj_name in obj_names}
        env_task_set[episode_id]["task_goal"] = {0: colab_goal, 1: dict()}

        init_rooms = rng.choices(room_list, k=2)
        env_task_set[episode_id]["init_rooms"] = init_rooms

    env_task_set = [env_task_set[i] for i in filtered_ids]
    filtered_ids = list(range(len(env_task_set)))

    print("before filtering:", len(episode_ids))
    print(" after filtering:", len(filtered_ids))
    print(Counter([env_task_set[i]["task_name"] for i in filtered_ids]).most_common())
    return env_task_set, filtered_ids
