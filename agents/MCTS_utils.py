import copy
import multiprocessing as mp
from functools import partial

from agents.MCTS import Node


def find_heuristic(
    agent_id, char_index, unsatisfied, env_graph, simulator, object_target
):
    # find_{index}

    target = int(object_target.split("_")[-1])
    observations = simulator.get_observations(env_graph, char_index=char_index)
    id2node = {node["id"]: node for node in env_graph["nodes"]}
    containerdict = {}
    hold = False
    for edge in env_graph["edges"]:
        if edge["relation_type"] == "INSIDE" and edge["from_id"] not in containerdict:
            containerdict[edge["from_id"]] = edge["to_id"]
        elif "HOLDS" in edge["relation_type"] and agent_id == 1:  # only for main agent
            containerdict[edge["to_id"]] = edge["from_id"]
            if edge["to_id"] == target:
                hold = True

    # containerdict = {
    #     edge['from_id']: edge['to_id']
    #     for edge in env_graph['edges']
    #     if edge['relation_type'] == 'INSIDE'
    # }

    observation_ids = [x["id"] for x in observations["nodes"]]

    # if agent_id == 1 and hold:
    #     print('container_ids find:', object_target, containerdict)

    try:
        room_char = [
            edge["to_id"]
            for edge in env_graph["edges"]
            if edge["from_id"] == agent_id and edge["relation_type"] == "INSIDE"
        ][0]
    except:
        print("Error")
        # ipdb.set_trace()

    action_list = []
    cost_list = []
    # if target == 478:
    #     )ipdb.set_trace()
    while target not in observation_ids:
        try:
            container = containerdict[target]
        except:
            print(id2node[target])
            raise Exception
        # If the object is a room, we have to walk to what is insde

        if id2node[container]["category"] == "Rooms":
            action_list = [
                ("walk", (id2node[target]["class_name"], target), None)
            ] + action_list
            cost_list = [0.5] + cost_list

        elif "CLOSED" in id2node[container]["states"] or (
            "OPEN" not in id2node[container]["states"]
        ):
            if id2node[container]["class_name"] != "character":
                action = ("open", (id2node[container]["class_name"], container), None)
                action_list = [action] + action_list
                cost_list = [0.05] + cost_list

        target = container
        # if hold:
        #     print(target)

    ids_character = [
        x["to_id"]
        for x in observations["edges"]
        if x["from_id"] == agent_id and x["relation_type"] == "CLOSE"
    ] + [
        x["from_id"]
        for x in observations["edges"]
        if x["to_id"] == agent_id and x["relation_type"] == "CLOSE"
    ]

    if target not in ids_character:
        # If character is not next to the object, walk there
        action_list = [
            ("walk", (id2node[target]["class_name"], target), None)
        ] + action_list
        cost_list = [1] + cost_list

    # if hold:
    #     print(action_list)

    return action_list, cost_list, f"find_{target}"


def touch_heuristic(
    agent_id, char_index, unsatisfied, env_graph, simulator, object_target
):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split("_")[-1])

    observed_ids = [node["id"] for node in observations["nodes"]]
    agent_close = [
        edge
        for edge in env_graph["edges"]
        if (
            (edge["from_id"] == agent_id and edge["to_id"] == target_id)
            or (edge["from_id"] == target_id and edge["to_id"] == agent_id)
            and edge["relation_type"] == "CLOSE"
        )
    ]

    target_node = [node for node in env_graph["nodes"] if node["id"] == target_id][0]

    target_action = [("touch", (target_node["class_name"], target_id), None)]
    cost = [0.05]

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost, f"touch_{target_id}"
    else:
        find_actions, find_costs, _ = find_heuristic(
            agent_id, char_index, unsatisfied, env_graph, simulator, object_target
        )
        return find_actions + target_action, find_costs + cost, f"touch_{target_id}"


def grab_heuristic(
    agent_id, char_index, unsatisfied, env_graph, simulator, object_target
):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split("_")[-1])

    observed_ids = [node["id"] for node in observations["nodes"]]
    agent_close = [
        edge
        for edge in env_graph["edges"]
        if (
            (edge["from_id"] == agent_id and edge["to_id"] == target_id)
            or (edge["from_id"] == target_id and edge["to_id"] == agent_id)
            and edge["relation_type"] == "CLOSE"
        )
    ]
    grabbed_obj_ids = [
        edge["to_id"]
        for edge in env_graph["edges"]
        if (edge["from_id"] == agent_id and "HOLDS" in edge["relation_type"])
    ]

    target_node = [node for node in env_graph["nodes"] if node["id"] == target_id][0]

    if target_id not in grabbed_obj_ids:
        target_action = [("grab", (target_node["class_name"], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    # if agent_id == 1:
    #     print('observed_ids grab:', target_id, observed_ids)

    if len(agent_close) > 0 and target_id in observed_ids:
        if agent_id == 1 and target_id == 351:
            print(target_action)
        return target_action, cost, f"grab_{target_id}"
    else:
        find_actions, find_costs, _ = find_heuristic(
            agent_id, char_index, unsatisfied, env_graph, simulator, object_target
        )
        if agent_id == 1 and target_id == 351:
            print(find_actions + target_action)
        return find_actions + target_action, find_costs + cost, f"grab_{target_id}"


def turnOn_heuristic(
    agent_id, char_index, unsatisfied, env_graph, simulator, object_target
):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split("_")[-1])

    observed_ids = [node["id"] for node in observations["nodes"]]
    agent_close = [
        edge
        for edge in env_graph["edges"]
        if (
            (edge["from_id"] == agent_id and edge["to_id"] == target_id)
            or (edge["from_id"] == target_id and edge["to_id"] == agent_id)
            and edge["relation_type"] == "CLOSE"
        )
    ]
    grabbed_obj_ids = [
        edge["to_id"]
        for edge in env_graph["edges"]
        if (edge["from_id"] == agent_id and "HOLDS" in edge["relation_type"])
    ]

    target_node = [node for node in env_graph["nodes"] if node["id"] == target_id][0]

    if target_id not in grabbed_obj_ids:
        target_action = [("switchon", (target_node["class_name"], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost, f"turnon_{target_id}"
    else:
        find_actions, find_costs = find_heuristic(
            agent_id, char_index, unsatisfied, env_graph, simulator, object_target
        )
        return find_actions + target_action, find_costs + cost, f"turnon_{target_id}"


def sit_heuristic(
    agent_id, char_index, unsatisfied, env_graph, simulator, object_target
):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split("_")[-1])

    observed_ids = [node["id"] for node in observations["nodes"]]
    agent_close = [
        edge
        for edge in env_graph["edges"]
        if (
            (edge["from_id"] == agent_id and edge["to_id"] == target_id)
            or (edge["from_id"] == target_id and edge["to_id"] == agent_id)
            and edge["relation_type"] == "CLOSE"
        )
    ]
    on_ids = [
        edge["to_id"]
        for edge in env_graph["edges"]
        if (edge["from_id"] == agent_id and "ON" in edge["relation_type"])
    ]

    target_node = [node for node in env_graph["nodes"] if node["id"] == target_id][0]

    if target_id not in on_ids:
        target_action = [("sit", (target_node["class_name"], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost, f"sit_{target_id}"
    else:
        find_actions, find_costs = find_heuristic(
            agent_id, char_index, unsatisfied, env_graph, simulator, object_target
        )
        return find_actions + target_action, find_costs + cost, f"sit_{target_id}"


def put_heuristic(
    agent_id, char_index, unsatisfied, env_graph, simulator, target, verbose=False
):
    # Modif, now put heristic is only the immaediate after action
    observations = simulator.get_observations(env_graph, char_index=char_index)

    target_grab, target_put = [int(x) for x in target.split("_")[-2:]]
    if verbose:
        raise AssertionError
    if (
        sum(
            [
                1
                for edge in observations["edges"]
                if edge["from_id"] == target_grab
                and edge["to_id"] == target_put
                and edge["relation_type"] == "ON"
            ]
        )
        > 0
    ):
        # Object has been placed
        # ipdb.set_trace()
        return [], 0, []

    if (
        sum(
            [
                1
                for edge in observations["edges"]
                if edge["to_id"] == target_grab
                and edge["from_id"] != agent_id
                and agent_id == 2
                and "HOLD" in edge["relation_type"]
            ]
        )
        > 0
    ):
        # Object is being placed by another agent
        # ipdb.set_trace()
        return [], 0, []

    target_node = [node for node in env_graph["nodes"] if node["id"] == target_grab][0]
    target_node2 = [node for node in env_graph["nodes"] if node["id"] == target_put][0]
    id2node = {node["id"]: node for node in env_graph["nodes"]}
    target_grabbed = (
        len(
            [
                edge
                for edge in env_graph["edges"]
                if edge["from_id"] == agent_id
                and "HOLDS" in edge["relation_type"]
                and edge["to_id"] == target_grab
            ]
        )
        > 0
    )
    if verbose:
        raise AssertionError
    object_diff_room = None
    if not target_grabbed:
        grab_obj1, cost_grab_obj1, heuristic_name = grab_heuristic(
            agent_id,
            char_index,
            unsatisfied,
            env_graph,
            simulator,
            "grab_" + str(target_node["id"]),
        )
        if len(grab_obj1) > 0:
            if grab_obj1[0][0] == "walk":
                id_room = grab_obj1[0][1][1]
                if id2node[id_room]["category"] == "Rooms":
                    object_diff_room = id_room

        return grab_obj1, cost_grab_obj1, heuristic_name
    else:
        env_graph_new = env_graph
        grab_obj1 = []
        cost_grab_obj1 = []
        find_obj2, cost_find_obj2, _ = find_heuristic(
            agent_id,
            char_index,
            unsatisfied,
            env_graph_new,
            simulator,
            "find_" + str(target_node2["id"]),
        )

    res = grab_obj1 + find_obj2
    cost_list = cost_grab_obj1 + cost_find_obj2

    if verbose:
        raise AssertionError
    if target_put > 2:  # not character
        action = [
            (
                "putback",
                (target_node["class_name"], target_grab),
                (target_node2["class_name"], target_put),
            )
        ]
        cost = [0.05]
        res += action
        cost_list += cost
    else:
        action = [("walk", (target_node2["class_name"], target_put), None)]
        cost = [0]
        res += action
        cost_list += cost
    # print(res, target)
    return res, cost_list, f"put_{target_grab}_{target_put}"


def putIn_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, target):
    # TODO: change this as well
    observations = simulator.get_observations(env_graph, char_index=char_index)

    target_grab, target_put = [int(x) for x in target.split("_")[-2:]]

    if (
        sum(
            [
                1
                for edge in observations["edges"]
                if edge["from_id"] == target_grab
                and edge["to_id"] == target_put
                and edge["relation_type"] == "ON"
            ]
        )
        > 0
    ):
        # Object has been placed
        return [], 0, []

    if (
        sum(
            [
                1
                for edge in observations["edges"]
                if edge["to_id"] == target_grab
                and edge["from_id"] != agent_id
                and agent_id == 2
                and "HOLD" in edge["relation_type"]
            ]
        )
        > 0
    ):
        # Object has been placed
        return None, 0, None

    target_node = [node for node in env_graph["nodes"] if node["id"] == target_grab][0]
    target_node2 = [node for node in env_graph["nodes"] if node["id"] == target_put][0]
    id2node = {node["id"]: node for node in env_graph["nodes"]}
    target_grabbed = (
        len(
            [
                edge
                for edge in env_graph["edges"]
                if edge["from_id"] == agent_id
                and "HOLDS" in edge["relation_type"]
                and edge["to_id"] == target_grab
            ]
        )
        > 0
    )

    object_diff_room = None
    if not target_grabbed:
        grab_obj1, cost_grab_obj1, heuristic_name = grab_heuristic(
            agent_id,
            char_index,
            unsatisfied,
            env_graph,
            simulator,
            "grab_" + str(target_node["id"]),
        )
        if len(grab_obj1) > 0:
            if grab_obj1[0][0] == "walk":
                id_room = grab_obj1[0][1][1]
                if id2node[id_room]["category"] == "Rooms":
                    object_diff_room = id_room

        return grab_obj1, cost_grab_obj1, heuristic_name

    else:
        grab_obj1 = []
        cost_grab_obj1 = []

        env_graph_new = env_graph
        grab_obj1 = []
        cost_grab_obj1 = []
        find_obj2, cost_find_obj2, _ = find_heuristic(
            agent_id,
            char_index,
            unsatisfied,
            env_graph_new,
            simulator,
            "find_" + str(target_node2["id"]),
        )
        target_put_state = target_node2["states"]
        action_open = [("open", (target_node2["class_name"], target_put))]
        action_put = [
            (
                "putin",
                (target_node["class_name"], target_grab),
                (target_node2["class_name"], target_put),
            )
        ]
        cost_open = [0.05]
        cost_put = [0.05]

        remained_to_put = 0
        for predicate, count in unsatisfied.items():
            if predicate.startswith("inside"):
                remained_to_put += count
        if remained_to_put == 1:  # or agent_id > 1:
            action_close = []
            cost_close = []
        else:
            action_close = []
            cost_close = []
            # action_close = [('close', (target_node2['class_name'], target_put))]
            # cost_close = [0.05]

        if "CLOSED" in target_put_state or "OPEN" not in target_put_state:
            res = grab_obj1 + find_obj2 + action_open + action_put + action_close
            cost_list = (
                cost_grab_obj1 + cost_find_obj2 + cost_open + cost_put + cost_close
            )
        else:
            res = grab_obj1 + find_obj2 + action_put + action_close
            cost_list = cost_grab_obj1 + cost_find_obj2 + cost_put + cost_close

        # print(res, target)
        grab_node = target_node["id"]
        place_node = target_node2["id"]
        return res, cost_list, f"putin_{grab_node}_{place_node}"


def clean_graph(state, goal_spec, last_opened):
    # TODO: document well what this is doing
    new_graph = {}
    # get all ids
    ids_interaction = []
    nodes_missing = []
    for predicate, val_goal in goal_spec.items():
        elements = predicate.split("_")
        nodes_missing += val_goal["grab_obj_ids"]
        nodes_missing += val_goal["container_ids"]
    # get all grabbed object ids
    for edge in state["edges"]:
        if "HOLD" in edge["relation_type"] and edge["to_id"] not in nodes_missing:
            nodes_missing += [edge["to_id"]]

    nodes_missing += [
        node["id"]
        for node in state["nodes"]
        if node["class_name"] == "character" or node["category"] in ["Rooms", "Doors"]
    ]

    def clean_node(curr_node):
        return {
            "id": curr_node["id"],
            "class_name": curr_node["class_name"],
            "category": curr_node["category"],
            "states": curr_node["states"],
            "properties": curr_node["properties"],
        }

    id2node = {node["id"]: clean_node(node) for node in state["nodes"]}
    # print([node for node in state['nodes'] if node['class_name'] == 'kitchentable'])
    # print(id2node[235])
    # ipdb.set_trace()
    inside = {}
    for edge in state["edges"]:
        if edge["relation_type"] == "INSIDE":
            if edge["from_id"] not in inside.keys():
                inside[edge["from_id"]] = []
            inside[edge["from_id"]].append(edge["to_id"])

    while len(nodes_missing) > 0:
        new_nodes_missing = []
        for node_missing in nodes_missing:
            if node_missing in inside:
                new_nodes_missing += [
                    node_in
                    for node_in in inside[node_missing]
                    if node_in not in ids_interaction
                ]
            ids_interaction.append(node_missing)
        nodes_missing = list(set(new_nodes_missing))

    if last_opened is not None:
        obj_id = int(last_opened[1][1:-1])
        if obj_id not in ids_interaction:
            ids_interaction.append(obj_id)

    # for clean up tasks, add places to put objects to
    augmented_class_names = []
    for key, value in goal_spec.items():
        elements = key.split("_")
        if elements[0] == "off":
            if id2node[value["containers"][0]]["class_name"] in [
                "dishwasher",
                "kitchentable",
            ]:
                augmented_class_names += [
                    "kitchencabinets",
                    "kitchencounterdrawer",
                    "kitchencounter",
                ]
                break
    for key in goal_spec:
        elements = key.split("_")
        if elements[0] == "off":
            if id2node[value["container_ids"][0]]["class_name"] in ["sofa", "chair"]:
                augmented_class_names += ["coffeetable"]
                break
    containers = [
        [node["id"], node["class_name"]]
        for node in state["nodes"]
        if node["class_name"] in augmented_class_names
    ]
    for obj_id in containers:
        if obj_id not in ids_interaction:
            ids_interaction.append(obj_id)

    new_graph = {
        "edges": [
            edge
            for edge in state["edges"]
            if edge["from_id"] in ids_interaction and edge["to_id"] in ids_interaction
        ],
        "nodes": [id2node[id_node] for id_node in ids_interaction],
    }

    return new_graph


def mp_run_mcts(root_node, mcts, nb_steps, last_subgoal, opponent_subgoal):
    heuristic_dict = {
        "offer": put_heuristic,
        "find": find_heuristic,
        "grab": grab_heuristic,
        "put": put_heuristic,
        "putIn": putIn_heuristic,
        "sit": sit_heuristic,
        "turnOn": turnOn_heuristic,
        "touch": touch_heuristic,
    }
    new_mcts = copy.deepcopy(mcts)
    res = new_mcts.run(
        root_node, nb_steps, heuristic_dict, last_subgoal, opponent_subgoal
    )
    return res


def mp_run_2(
    process_id, root_node, mcts, nb_steps, last_subgoal, opponent_subgoal, res
):
    res[process_id] = mp_run_mcts(
        root_node=root_node,
        mcts=mcts,
        nb_steps=nb_steps,
        last_subgoal=last_subgoal,
        opponent_subgoal=opponent_subgoal,
    )


def get_plan(
    mcts,
    particles,
    env,
    nb_steps,
    goal_spec,
    last_subgoal,
    last_action,
    opponent_subgoal=None,
    num_process=10,
    length_plan=5,
    verbose=True,
):
    root_nodes = []
    for particle_id in range(len(particles)):
        root_action = None
        root_node = Node(
            id=(root_action, [goal_spec, 0, ""]),
            particle_id=particle_id,
            plan=[],
            state=copy.deepcopy(particles[particle_id]),
            num_visited=0,
            sum_value=0,
            is_expanded=False,
        )
        root_nodes.append(root_node)

    # root_nodes = list(range(10))
    mp_run = partial(
        mp_run_mcts,
        mcts=mcts,
        nb_steps=nb_steps,
        last_subgoal=last_subgoal,
        opponent_subgoal=opponent_subgoal,
    )

    if len(root_nodes) == 0:
        print("No root nodes")
        raise Exception
    if num_process > 0:
        manager = mp.Manager()
        res = manager.dict()
        num_root_nodes = len(root_nodes)
        for start_root_id in range(0, num_root_nodes, num_process):
            end_root_id = min(start_root_id + num_process, num_root_nodes)
            jobs = []
            for process_id in range(start_root_id, end_root_id):
                # print(process_id)
                p = mp.Process(
                    target=mp_run_2,
                    args=(
                        process_id,
                        root_nodes[process_id],
                        mcts,
                        nb_steps,
                        last_subgoal,
                        opponent_subgoal,
                        res,
                    ),
                )
                jobs.append(p)
                p.start()
            for p in jobs:
                p.join()
        info = [res[x] for x in range(len(root_nodes))]

    else:
        info = [mp_run(rn) for rn in root_nodes]

    for info_item in info:
        if not isinstance(info_item, tuple):
            print("rasiing")
            info_item.re_raise()

    if num_process > 0:
        print("Plan Done")
    rewards_all = [inf[-1] for inf in info]
    plans_all = [inf[1] for inf in info]
    goals_all = [inf[-2] for inf in info]
    index_action = 0
    # length_plan = 5
    prev_index_particles = list(range(len(info)))

    final_actions, final_goals = [], []
    lambd = 0.5
    # ipdb.set_trace()
    while index_action < length_plan:
        max_action = None
        max_score = None
        action_count_dict = {}
        action_reward_dict = {}
        action_goal_dict = {}
        # Which particles we select now
        index_particles = [
            p_id for p_id in prev_index_particles if len(plans_all[p_id]) > index_action
        ]
        # print(index_particles)
        if len(index_particles) == 0:
            index_action += 1
            continue
        for ind in index_particles:
            action = plans_all[ind][index_action]
            if action is None:
                continue
            try:
                reward = rewards_all[ind][index_action]
                goal = goals_all[ind][index_action]
            except:
                raise AssertionError
            if action not in action_count_dict:
                action_count_dict[action] = []
                action_goal_dict[action] = []
                action_reward_dict[action] = 0
            action_count_dict[action].append(ind)
            action_reward_dict[action] += reward
            action_goal_dict[action].append(goal)

        for action in action_count_dict:
            # Average reward of this action
            average_reward = (
                action_reward_dict[action] * 1.0 / len(action_count_dict[action])
            )
            # Average proportion of particles
            average_visit = len(action_count_dict[action]) * 1.0 / len(index_particles)
            score = average_reward * lambd + average_visit
            goal = action_goal_dict[action]

            if max_score is None or max_score < score:
                max_score = score
                max_action = action
                max_goal = goal

        index_action += 1
        prev_index_particles = action_count_dict[max_action]
        # print(max_action, prev_index_particles)
        final_actions.append(max_action)
        final_goals.append(max_goal)

    # ipdb.set_trace()
    # If there is no action predicted but there were goals missing...
    # if len(final_actions) == 0:
    #     print("No actions")
    if verbose:
        raise AssertionError

    plan = final_actions
    subgoals = final_goals

    # ipdb.set_trace()
    # subgoals = [[None, None, None], [None, None, None]]
    # next_root, plan, subgoals = mp_run_mcts(root_nodes[0])
    next_root = None

    # ipdb.set_trace()
    # print('plan', plan)
    # if 'put' in plan[0]:
    #     ipdb.set_trace()
    if verbose:
        print("plan", plan)
        print("subgoal", subgoals)
    sample_id = None

    if sample_id is not None:
        res[sample_id] = plan
    else:
        return plan, next_root, subgoals
