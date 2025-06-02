import re
from collections import Counter
from dataclasses import dataclass

from virtualhome.simulation.evolving_graph.environment import (
    Bounds,
    EnvironmentGraph,
    GraphEdge,
    GraphNode,
    Property,
    Relation,
    State,
)

from utils import utils_environment as utils_env
from utils.utils_logging import format_row, get_my_logger


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

def parse_string(s):
    # Define a regular expression pattern to match the parts of the string
    pattern = r"\[(.*?)\] <(.*?)> \((\d+)\) <(.*?)> \((\d+)\)"
    match = re.match(pattern, s)

    if match:
        # Extract the parts and convert the number to an integer
        action = match.group(1)
        device = match.group(2)
        number = match.group(3)
        device1 = match.group(4)
        number1 = match.group(5)
        return [action, device, number, device1, number1]

    else:
        # If the pattern does not match, return an empty list or raise an error
        pattern1 = r"\[(.*?)\] <(.*?)> \((\d+)\)"
        match = re.match(pattern1, s)
        if match:
            action = match.group(1)
            device = match.group(2)
            number = match.group(3)
            return [action, device, number]


class GN(GraphNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indeg = []
        self.outdeg = []

    @staticmethod
    def from_dict(d):
        kwargs = {"category": None, "prefab_name": None, "bounding_box": None}
        for k in kwargs.keys():
            if k in d:
                if k == "bounding_box":
                    kwargs[k] = Bounds(**d[k]) if d[k] is not None else d[k]
                else:
                    kwargs[k] = d[k]

        return GN(
            d["id"],
            d["class_name"],
            {
                s if isinstance(s, Property) else Property[s.upper()]
                for s in d["properties"]
            },
            {State[s.upper()] for s in d["states"]},
            **kwargs,
        )

    def __str__(self):
        basic = f"{self.id:3d}|{self.class_name}"
        # if Property.CONTAINERS in self.properties:
        if len(self.states) > 0:
            states = [s.name for s in self.states]
            basic += f"|{'+'.join(states)}"
        return basic

    def __repr__(self):
        return self.__str__()

    def filter_deg(self, relation, deg_type):
        return [
            self.graph[node_id]
            for node_id, rel in getattr(self, deg_type)
            if rel == relation
        ]

    def contained_by(self):
        return self.filter_deg(Relation.INSIDE, "outdeg")

    def contains(self):
        return self.filter_deg(Relation.INSIDE, "indeg")

    def supported_by(self):
        return self.filter_deg(Relation.ON, "outdeg")

    def supports(self):
        return self.filter_deg(Relation.ON, "indeg")

    def holds(self):
        lh = self.filter_deg(Relation.HOLDS_LH, "outdeg")
        rh = self.filter_deg(Relation.HOLDS_RH, "outdeg")
        return lh + rh

    def held_by(self):
        lh = self.filter_deg(Relation.HOLDS_LH, "indeg")
        rh = self.filter_deg(Relation.HOLDS_RH, "indeg")
        return lh + rh

    def close(self):
        idg = self.filter_deg(Relation.CLOSE, "indeg")
        odg = self.filter_deg(Relation.CLOSE, "outdeg")
        assert {n.id for n in idg} == {n.id for n in odg}, (idg, odg)
        return idg

    @staticmethod
    def _assure_unique(elems, fix):
        elems = set(elems)
        match len(elems):
            case 0:
                return None
            case 1:
                return elems.pop()
            case _:
                if fix:
                    return elems
                else:
                    raise AssertionError(elems)

    def get_room(self, fix=False):
        rooms = []
        for obj in self.contained_by():
            if obj.category == "Rooms":
                rooms.append(obj)
            else:
                ctnrs = obj.contained_by()
                assert all(ctnr.category == "Rooms" for ctnr in ctnrs)
                rooms.extend(ctnrs)
        return self._assure_unique(rooms, fix)

    def get_ctnr(self, fix=False):
        ctnrs = []
        for obj in self.contained_by():
            if obj.category == "Rooms":
                continue
            else:
                ctnrs.append(obj)
        return self._assure_unique(ctnrs, fix)

    def get_srfc(self, fix=False):
        srfc = self.supported_by()
        srfc = [x for x in srfc if x.class_name not in ("floor", "rug")]
        return self._assure_unique(srfc, fix)

    def get_location(self, fix=False):
        room = self.get_room(fix)
        ctnr = self.get_ctnr(fix)
        srfc = self.get_srfc(fix)
        return room, ctnr, srfc

    @staticmethod
    def res(n):
        return (None, None) if n is None else (n.id, n.class_name)

    def __format__(self, format_spec):
        return format(str(self), format_spec)


class GE(GraphEdge):
    def __str__(self):
        return str((self.from_node, self.relation, self.to_node))

    def __repr__(self):
        return self.__str__()


@dataclass
class Subgoal:
    name: str
    cnt: int
    obj_nodes: list[GN]
    tgt_nodes: list[GN]

    def __post_init__(self):
        self.prep = self.name.split("_")[0]

    def __str__(self):
        return f"{self.tgt_nodes} <== {self.cnt} of {self.obj_nodes}"

    def __repr__(self):
        return self.__str__()

    @property
    def natlang(self):
        obj_name = self.obj_nodes[0].class_name
        tgt_name = self.tgt_nodes[0].class_name
        return f"Put {self.cnt} {obj_name} {self.prep} the {tgt_name}"


class Goal:
    def __init__(self, goal, eg):
        # * convert `goal` to `goal_spec`
        if isinstance(list(goal.values())[0], int):
            goal = utils_env.convert_goal(goal, eg._dictionary)

        self.eg = eg
        self.subgoals = []
        for subgoal_name, subgoal in goal.items():
            name = subgoal_name
            cnt = subgoal["count"]
            obj_ids = subgoal["grab_obj_ids"]
            tgt_ids = subgoal["container_ids"]
            obj_nodes = [eg[obj_id] for obj_id in obj_ids]
            tgt_nodes = [eg[tgt_id] for tgt_id in tgt_ids]
            self.subgoals.append(Subgoal(name, cnt, obj_nodes, tgt_nodes))

    def __str__(self):
        return "\n".join([str(subgoal) for subgoal in self.subgoals])

    def __repr__(self):
        return self.__str__()

    @property
    def natlang(self):
        return "\n".join([f"[{i}] {sg.natlang}" for i, sg in enumerate(self.subgoals)])


class EG(EnvironmentGraph):
    def __init__(self, dictionary):
        super().__init__(dictionary)

        self.lg = get_my_logger()
        self.lg_res = get_my_logger(name="shunchi_res_py")
        self._dictionary = dictionary

    def _from_dictionary(self, d):
        nodes = [GN.from_dict(n) for n in d["nodes"]]
        for n in nodes:
            self._node_map[n.id] = n
            self._class_name_map.setdefault(n.class_name, []).append(n)
            if n.id > self._max_node_id:
                self._max_node_id = n.id

        for ed in d["edges"]:
            from_id = ed["from_id"]
            relation = Relation[ed["relation_type"].upper()]
            to_id = ed["to_id"]

            es = self._edge_map.setdefault((from_id, relation), {})
            es[to_id] = self._node_map[to_id]

            self._node_map[from_id].outdeg.append([to_id, relation])
            self._node_map[to_id].indeg.append([from_id, relation])

        for node_id in self._node_map.keys():
            self._node_map[node_id].graph = self

    def __getitem__(self, node_id):
        return self._node_map[node_id]

    def filter_nodes(self, filter):
        return [node for node in self._node_map.values() if filter(node)]

    def has_class(self, class_name):
        return len(list(self.get_nodes_by_attr("class_name", class_name))) > 0

    def inspect(self):
        n_nodes = len(self.get_nodes())
        n_edges = len(self.get_from_pairs())
        agents = self.get_agents()
        agent_ids = [a.id for a in agents]
        return f"#nodes = {n_nodes}, #edges = {n_edges}, agents = {agent_ids}"

    def get_rooms(self):
        rooms = self.get_nodes_by_attr("category", "Rooms")
        return list(rooms)

    def get_containers(self):
        containers = self.filter_nodes(lambda x: Property.CONTAINERS in x.properties)
        return list(containers)

    def get_surfaces(self):
        surfaces = self.filter_nodes(lambda x: Property.SURFACES in x.properties)
        return list(surfaces)

    def get_agents(self):
        agents = self.get_nodes_by_attr("class_name", "character")
        return list(agents)

    def goal_table(self, goal, title):
        assert self.lg is not None
        # raise NotImplementedError("node.get_location() bug")
        log_content = format(title, "-^120")
        self.lg.info(log_content)
        self.lg_res.debug(log_content)
        log_content = format_row(["Object", "Room", "Container", "Surface"], "<30")
        self.lg.debug(log_content)
        self.lg_res.debug(log_content)

        ret = dict()

        for subgoal_name, subgoal_spec in goal.items():
            log_content = format(subgoal_name, ".^30")
            self.lg.debug(log_content)
            self.lg_res.debug(log_content)
            obj_ids = subgoal_spec["grab_obj_ids"]
            tgt_ids = subgoal_spec["container_ids"]

            for obj_id in obj_ids:
                obj = self[obj_id]
                if obj is None:
                    log_content = format_row([obj_id] + ["Unknown"] * 3, "<30")
                else:
                    room, ctnr, srfc = obj.get_location()
                    ret[GN.res(obj)] = dict(
                        room=GN.res(room),
                        ctnr=GN.res(ctnr),
                        srfc=GN.res(srfc),
                    )
                    log_content = format_row([obj, room, ctnr, srfc], "<30")
                    self.lg_res.debug(log_content)
                self.lg.debug(log_content)

            for tgt_id in tgt_ids:
                tgt = self[tgt_id]
                if tgt is None:
                    log_content = format_row([tgt_id] + ["Unknown"] * 3, "<30")
                else:
                    room, ctnr, srfc = tgt.get_location()
                    ret[GN.res(tgt)] = dict(
                        room=GN.res(room),
                        ctnr=GN.res(ctnr),
                        srfc=GN.res(srfc),
                    )
                    log_content = format_row([tgt, room, ctnr, srfc], "<30")
                    self.lg_res.debug(log_content)
                self.lg.debug(log_content)

        # self.lg.info(format(" Describe Done ", "=^120") + "\n")

        return ret

    def actions_to_natlang(self, actions, init_room, name="Human"):
        lines = [f"{name} is in the {init_room}"]
        for action in actions:
            if action is None:
                continue
            parsed = parse_string(action)
            predicate = parsed[0]
            match predicate:
                case "walk":
                    line = f"{name} walks to the {parsed[1]}"
                case "putback":
                    prep = "on"
                    line = f"{name} puts the {parsed[1]} {prep} the {parsed[3]}"
                case "putin":
                    prep = "in"
                    line = f"{name} puts the {parsed[1]} {prep} the {parsed[3]}"
                case "open" | "grab":
                    line = f"{name} {predicate}s the {parsed[1]}"
                case _:
                    raise ValueError(parsed)
            lines.append(line)
        # return "\n".join([f"[{i}] {line}" for i, line in enumerate(lines)])
        return lines

    def story(self, ctnr_ids, srfc_ids):
        """
        >>>
        code for restricting the story to TGT_LIST and OBJ_SET
        <<<
        """
        tree = dict()  # room => ctnr|srfc => obj

        rooms = self.get_rooms()
        for room in rooms:
            tree[room.class_name] = []

        ctnr_set = set(ctnr_ids)
        srfc_set = set(srfc_ids)

        # & >>>>>
        # ctnr_set = set()
        # srfc_set = set()
        # for predicate, tgt_id in TGT_LIST:
        #     match predicate:
        #         case "on":
        #             srfc_set.add(tgt_id)
        #         case "inside":
        #             ctnr_set.add(tgt_id)
        #         case _:
        #             raise ValueError(predicate)
        # & <<<<<

        for mid_id in ctnr_set | srfc_set:
            if mid_id in ctnr_set:
                predicate = "contains"
            elif mid_id in srfc_set:
                predicate = "supports"
            else:
                raise ValueError(mid_id)

            mid = self[mid_id]
            mid_name = mid.class_name
            room = mid.get_room()
            room_name = room.class_name

            objs = Counter([n.class_name for n in getattr(mid, predicate)()])
            objs = [f"{cnt} {obj}" for obj, cnt in objs.items() if obj]
            # & >>>>>
            objs = [o for o in objs if o.split(" ")[-1] in OBJ_SET]
            # & <<<<<
            if len(objs) > 0:
                objs_str = ", ".join(objs)
                objs_str = f"The {mid_name} {predicate} {objs_str}."
            else:
                objs_str = ""

            tree[room_name].append([mid_name, objs_str])

        room_strs = []
        for room_name, room_info in tree.items():
            room_info = sorted(room_info)
            mid_names = Counter([mid_name for mid_name, _ in room_info])
            mid_names = [f"{cnt} {mid_name}" for mid_name, cnt in mid_names.items()]
            mid_names = ", ".join(mid_names)
            room_str = [f"The {room_name} has {mid_names}. Specifically,"]
            room_str.extend(
                [f"- {mid_str}" for _, mid_str in room_info if mid_str != ""]
            )
            if len(room_str) >= 2:
                room_str = "\n".join(room_str)
                room_strs.append(room_str)
        room_strs = "\n\n".join(room_strs)

        story = "The apartment has {} rooms: {}.\n\n{}".format(
            len(rooms),
            ", ".join([r.class_name for r in rooms]),
            room_strs,
        )

        return story

    def agent_state_natlang(self, agent_id=1, name="Human"):
        agent = self[agent_id]
        close_objs = agent.close()
        in_hands = agent.holds()
        room_name = agent.get_room().class_name
        lines = [f"{name} is in the {room_name}."]
        if len(close_objs) > 0:
            close_objs = Counter([o.class_name for o in close_objs])
            lines.append(
                f"{name} is close to {', '.join(f'{cnt} {obj}' for obj, cnt in close_objs.items())}."
            )
        if len(in_hands) > 0:
            in_hands = Counter([o.class_name for o in in_hands])
            lines.append(
                f"{name} is holding {', '.join(f'{cnt} {obj}' for obj, cnt in in_hands.items())}."
            )
        return "\n".join(lines)

    def goal_rooms(self, goal):
        obj_rooms = []
        tgt_rooms = []

        goal = Goal(goal, self)
        obj_rooms = [n.get_room() for sg in goal.subgoals for n in sg.obj_nodes]
        tgt_rooms = [n.get_room() for sg in goal.subgoals for n in sg.tgt_nodes]

        obj_rooms = [room for room, cnt in Counter(obj_rooms).most_common()]
        tgt_rooms = [room for room, cnt in Counter(tgt_rooms).most_common()]

        return obj_rooms, tgt_rooms

    def goal_tree(self, goal, title):
        assert self.lg is not None
        self.lg.info(format(title, "-^120"))
        self.lg.info(format("objects", ".^30"))
        obj_nodes = []
        for subgoal_name, subgoal_spec in goal.items():
            obj_ids = subgoal_spec["grab_obj_ids"]
            for obj_id in obj_ids:
                obj_node = self[obj_id]
                obj_nodes.append(obj_node)

                self.lg.info(obj_node)
                in_objs = obj_node.contained_by()
                on_objs = obj_node.supported_by()
                if len(in_objs) > 0:
                    self.lg.debug(f"    <in> {in_objs} </in>")
                if len(on_objs) > 0:
                    self.lg.debug(f"    <on> {on_objs} </on>")

        self.lg.info(obj_nodes)
        # self.lg.info(format(" Describe Done ", "=^120") + "\n")

    def room_tree(self, title):
        raise NotImplementedError("Not implemented")
        assert self.lg is not None
        exclude_classes = ["floor", "wall", "ceiling", "window"]
        # * describe by room
        rooms = self.get_rooms()
        for room in rooms:
            self.lg.debug(format(str(room), ".^30"))
            xs = room.contains(self)
            for x in xs:
                if x.class_name in exclude_classes:
                    continue

                in_objs = [_ for _ in x.contains(self) if _ in obj_nodes]
                on_objs = [_ for _ in x.surface(self) if _ in obj_nodes]
                if len(in_objs) == len(on_objs) == 0:
                    continue
                self.lg.debug(x)
                if len(in_objs) > 0:
                    self.lg.debug(f"    <in> {in_objs} </in>")
                if len(on_objs) > 0:
                    self.lg.debug(f"    <on> {on_objs} </on>")

        self.lg.debug("Done Rooms")

        # * describe by agent
        agents = self.get_agents()
        for agent in agents:
            self.lg.debug(format(str(agent), ".^30"))


def fix_multiple_location(env_task_set, verbose=False, drop_env=False):
    """
    Only fix multiple location of goal objects,
    by simply keeping the first one.
    """

    def keep_first(edges, obj_id, relation_type, error_nodes):
        return [
            edge
            for edge in edges
            if not all(
                [
                    edge["from_id"] == obj_id,
                    edge["to_id"] in [x.id for x in error_nodes][1:],
                    edge["relation_type"] == relation_type,
                ]
            )
        ]

    bug_tasks = set()

    for ith_task, env_task in enumerate(env_task_set):
        eg = EG(env_task["init_graph"])
        edges = env_task["init_graph"]["edges"]
        human_goal = utils_env.convert_goal(
            env_task["task_goal"][0], env_task["init_graph"]
        )
        for subgoal_name, subgoal in human_goal.items():
            for obj_id in subgoal["grab_obj_ids"]:
                obj = eg[obj_id]
                room, ctnr, srfc = obj.get_location(fix=True)
                if isinstance(room, set):
                    if verbose:
                        print(f"ROOM [{ith_task:02d}] {obj:<20} {room}")
                    env_task["init_graph"]["edges"] = keep_first(
                        edges, obj_id, "INSIDE", room
                    )
                    bug_tasks.add(ith_task)
                if isinstance(ctnr, set):
                    if verbose:
                        print(f"CTNR [{ith_task:02d}] {obj:<20} {ctnr}")
                    env_task["init_graph"]["edges"] = keep_first(
                        edges, obj_id, "INSIDE", ctnr
                    )
                    bug_tasks.add(ith_task)
                if isinstance(srfc, set):
                    if verbose:
                        print(f"SRFC [{ith_task:02d}] {obj:<20} {srfc}")
                    env_task["init_graph"]["edges"] = keep_first(
                        edges, obj_id, "ON", srfc
                    )
                    bug_tasks.add(ith_task)

    if drop_env:
        if verbose:
            print(f"Dropping tasks: {bug_tasks}")
        env_task_set = [
            env_task for i, env_task in enumerate(env_task_set) if i not in bug_tasks
        ]

    if verbose:
        print(Counter([x["env_id"] for x in env_task_set]).most_common())
        print(Counter([x["task_name"] for x in env_task_set]).most_common())

    return env_task_set
