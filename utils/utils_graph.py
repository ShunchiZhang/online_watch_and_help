def fix_graph(env_task_set):
    for env in env_task_set:
        init_gr = env["init_graph"]
        gbg_can = [
            node["id"]
            for node in init_gr["nodes"]
            if node["class_name"] in ["garbagecan", "clothespile"]
        ]
        init_gr["nodes"] = [
            node for node in init_gr["nodes"] if node["id"] not in gbg_can
        ]
        init_gr["edges"] = [
            edge
            for edge in init_gr["edges"]
            if edge["from_id"] not in gbg_can and edge["to_id"] not in gbg_can
        ]
        for node in init_gr["nodes"]:
            if node["class_name"] == "cutleryfork":
                node["obj_transform"]["position"][1] += 0.1
    return env_task_set
