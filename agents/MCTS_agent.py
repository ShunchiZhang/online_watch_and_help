import copy
import time

from agents import MCTS, MCTS_utils, belief
from envs.graph_env import VhGraphEnv
from utils import utils_environment as utils_env


class MCTS_agent:
    """
    MCTS for a single agent
    """

    def __init__(
        self,
        agent_id,
        char_index,
        max_episode_length,
        num_simulation,
        max_rollout_steps,
        c_init,
        c_base,
        num_particles=20,
        recursive=False,
        num_samples=1,
        num_processes=1,
        comm=None,
        logging=False,
        logging_graphs=False,
        agent_params={},
        get_plan_states=False,
        get_plan_cost=False,
    ):
        self.agent_type = "MCTS"
        self.verbose = False
        self.recursive = recursive

        # self.env = unity_env.env
        self.logging = logging
        self.logging_graphs = logging_graphs

        self.last_obs = None
        self.last_plan = None
        self.last_loc = {}
        self.failed_action = False

        self.agent_id = agent_id
        self.char_index = char_index

        self.sim_env = VhGraphEnv(n_chars=self.agent_id)
        self.sim_env.pomdp = True
        self.belief = None

        self.belief_params = agent_params["belief"]
        self.agent_params = agent_params
        self.max_episode_length = max_episode_length
        self.num_simulation = num_simulation
        self.max_rollout_steps = max_rollout_steps
        self.c_init = c_init
        self.c_base = c_base
        self.num_samples = num_samples
        self.num_processes = num_processes
        self.num_particles = num_particles
        self.get_plan_states = get_plan_states
        self.get_plan_cost = get_plan_cost

        self.previous_belief_graph = None
        self.verbose = False

        # self.should_close = True
        # if self.planner_params:
        #     if 'should_close' in self.planner_params:
        #         self.should_close = self.planner_params['should_close']

        self.mcts = None
        # MCTS_particles_v2(self.agent_id, self.char_index, self.max_episode_length,
        #                 self.num_simulation, self.max_rollout_steps,
        #                 self.c_init, self.c_base, agent_params=self.agent_params)

        self.particles = [None for _ in range(self.num_particles)]
        self.particles_full = [None for _ in range(self.num_particles)]

        # if self.mcts is None:
        #    raise Exception

        # Indicates whether there is a unity simulation
        self.comm = comm

    def filtering_graph(self, graph):
        new_edges = []
        edge_dict = {}
        for edge in graph["edges"]:
            key = (edge["from_id"], edge["to_id"])
            if key not in edge_dict:
                edge_dict[key] = [edge["relation_type"]]
                new_edges.append(edge)
            else:
                if edge["relation_type"] not in edge_dict[key]:
                    edge_dict[key] += [edge["relation_type"]]
                    new_edges.append(edge)

        graph["edges"] = new_edges
        return graph

    def sample_belief(self, obs_graph):
        new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
        self.previous_belief_graph = self.filtering_graph(new_graph)
        return new_graph

    def get_relations_char(self, graph):
        # TODO: move this in vh_mdp
        char_id = [
            node["id"] for node in graph["nodes"] if node["class_name"] == "character"
        ][0]
        edges = [edge for edge in graph["edges"] if edge["from_id"] == char_id]
        print("Character:")
        print(edges)
        print("---")

    def get_location_in_goal(self, obs, obj_id):
        curr_loc = [edge for edge in obs["edges"] if edge["from_id"] == obj_id]
        curr_loc += [edge for edge in obs["edges"] if edge["to_id"] == obj_id]
        curr_loc_on = [
            edge
            for edge in curr_loc
            if edge["relation_type"] == "ON" or "hold" in edge["relation_type"].lower()
        ]
        curr_loc_inside = [
            edge for edge in curr_loc if edge["relation_type"] == "INSIDE"
        ]
        if len(curr_loc_on) + len(curr_loc_inside) > 0:
            if len(curr_loc_on) > 0:
                if "hold" in curr_loc_on[0]["relation_type"].lower():
                    curr_loc_index = curr_loc_on[0]["from_id"]
                else:
                    curr_loc_index = curr_loc_on[0]["to_id"]
            else:
                curr_loc_index = curr_loc_inside[0]["to_id"]
            if len(curr_loc_on) > 1 or len(curr_loc_inside) > 1:
                print(curr_loc_on)
                print(curr_loc_inside)
                raise AssertionError("Multiple locations for the same object")
            return curr_loc_index

    def get_action(
        self, obs, goal_spec, opponent_subgoal=None, length_plan=5, must_replan=False
    ):
        # ipdb.set_trace()
        if len(goal_spec) == 0:
            raise AssertionError

        # Create the particles
        # pdb.set_trace()

        self.belief.update_belief(obs)

        # TODO: maybe we will want to keep the previous belief graph to avoid replanning
        # self.sim_env.reset(self.previous_belief_graph, {0: goal_spec, 1: goal_spec})

        last_action = self.last_action
        last_subgoal = self.last_subgoal[0] if self.last_subgoal is not None else None
        subgoals = self.last_subgoal
        last_plan = self.last_plan

        # TODO: is this correct?
        nb_steps = 0
        root_action = None
        root_node = None
        verbose = self.verbose

        # If the current obs is the same as the last obs
        ignore_id = None

        should_replan = True

        goal_ids_all = []
        for goal_name, goal_val in goal_spec.items():
            if goal_val["count"] > 0:
                goal_ids_all += goal_val["grab_obj_ids"]

        goal_ids = [
            nodeobs["id"] for nodeobs in obs["nodes"] if nodeobs["id"] in goal_ids_all
        ]
        close_ids = [
            edge["to_id"]
            for edge in obs["edges"]
            if edge["from_id"] == self.agent_id
            and edge["relation_type"] in ["CLOSE", "INSIDE"]
        ]
        plan = []

        if last_plan is not None and len(last_plan) > 0:
            should_replan = False

            # If there is a goal object that was not there before
            next_id_interaction = []
            if len(last_plan) > 1:
                next_id_interaction.append(
                    int(last_plan[1].split("(")[1].split(")")[0])
                )

            new_observed_objects = (
                set(goal_ids)
                - set(self.last_obs["goal_objs"])
                - set(next_id_interaction)
            )
            # self.last_obs = {'goal_objs': goal_ids}
            if len(new_observed_objects) > 0:
                # New goal, need to replan
                should_replan = True
            else:
                visible_ids = {node["id"]: node for node in obs["nodes"]}
                curr_plan = last_plan

                first_action_non_walk = [
                    act for act in last_plan[1:] if "walk" not in act
                ]

                # If the first action other than walk is OPEN/CLOSE and the object is already open/closed...
                if len(first_action_non_walk):
                    first_action_non_walk = first_action_non_walk[0]
                    if "open" in first_action_non_walk:
                        obj_id = int(curr_plan[0].split("(")[1].split(")")[0])
                        if obj_id in visible_ids:
                            if "OPEN" in visible_ids[obj_id]["states"]:
                                should_replan = True
                                print("IS OPEN")
                    elif "close" in first_action_non_walk:
                        obj_id = int(curr_plan[0].split("(")[1].split(")")[0])
                        if obj_id in visible_ids:
                            if "CLOSED" in visible_ids[obj_id]["states"]:
                                should_replan = True
                                print("IS CLOSED")

                if (
                    "open" in last_plan[0]
                    or "close" in last_plan[0]
                    or "put" in last_plan[0]
                    or "grab" in last_plan[0]
                    or "touch" in last_plan[0]
                ):
                    if len(last_plan) == 1:
                        should_replan = True
                    else:
                        curr_plan = last_plan[1:]
                        subgoals = (
                            self.last_subgoal[1:]
                            if self.last_subgoal is not None
                            else None
                        )
                if (
                    "open" in curr_plan[0]
                    or "close" in curr_plan[0]
                    or "put" in curr_plan[0]
                    or "grab" in curr_plan[0]
                    or "touch" in curr_plan[0]
                ):
                    obj_id = int(curr_plan[0].split("(")[1].split(")")[0])
                    if obj_id not in close_ids or obj_id not in visible_ids:
                        should_replan = True

                next_action = not should_replan
                while next_action and "walk" in curr_plan[0]:
                    obj_id = int(curr_plan[0].split("(")[1].split(")")[0])

                    # If object is not visible, replan
                    if obj_id not in visible_ids:
                        should_replan = True
                        next_action = False
                    else:
                        if obj_id in close_ids:
                            if len(curr_plan) == 1:
                                should_replan = True
                                next_action = False
                            else:
                                curr_plan = curr_plan[1:]
                                subgoals = (
                                    subgoals[1:] if subgoals is not None else None
                                )
                        else:
                            # Keep with previous action
                            next_action = False

                if not should_replan:
                    plan = curr_plan

        obj_grab = -1
        curr_loc_index = -1

        self.last_obs = {"goal_objs": goal_ids}
        if self.failed_action:
            should_replan = True
            self.failed_action = False
        else:
            # obs = utils_env.inside_not_trans(obs)
            if not should_replan and not must_replan:
                # If the location of the object you wanted to grab has changed
                if last_subgoal is not None and "grab" in last_subgoal[0]:
                    obj_grab = int(last_subgoal[0].split("_")[1])
                    curr_loc_index = self.get_location_in_goal(obs, obj_grab)
                    if obj_grab not in self.last_loc:
                        raise AssertionError
                    if curr_loc_index != self.last_loc[obj_grab]:
                        # The object I wanted to get now changed position, so I should replan
                        # self.last_loc = curr_loc_index
                        should_replan = True

                # If you wanted to put an object but it is not in your hands anymore
                if last_subgoal is not None and "put" in last_subgoal[0]:
                    object_put = int(last_subgoal[0].split("_")[1])
                    hands_char = [
                        edge["to_id"]
                        for edge in obs["edges"]
                        if "hold" in edge["relation_type"].lower()
                        and edge["from_id"] == self.agent_id
                    ]
                    if object_put not in hands_char:
                        should_replan = True

        time1 = time.time()
        lg = ""
        if last_subgoal is not None and len(last_subgoal) > 0:
            lg = last_subgoal[0]
        if self.verbose:
            print(
                "-------- Agent {}: {} --------".format(
                    self.agent_id, "replan" if should_replan else "no replan"
                )
            )
        if should_replan or must_replan:
            # ipdb.set_trace()
            for particle_id, particle in enumerate(self.particles):
                belief_states = []
                obs_ids = [node["id"] for node in obs["nodes"]]

                # if True: #particle is None:
                new_graph = self.belief.update_graph_from_gt_graph(
                    obs, resample_unseen_nodes=True, update_belief=False
                )
                # print('new_graph:')
                # print([n['id'] for n in new_graph['nodes']])
                # print(
                #     'obs:',
                #     [edge for edge in obs['edges'] if 'HOLD' in edge['relation_type']],
                # )

                # print(
                #     'new_graph:',
                #     [
                #         edge
                #         for edge in new_graph['edges']
                #         if 'HOLD' in edge['relation_type']
                #     ],
                # )
                init_state = MCTS_utils.clean_graph(
                    new_graph, goal_spec, self.mcts.last_opened
                )
                satisfied, unsatisfied = utils_env.check_progress2(
                    init_state, goal_spec
                )
                if "offer" in list(goal_spec.keys())[0]:
                    if self.verbose:
                        print("offer:")
                        print(satisfied)
                        print(unsatisfied)
                    # ipdb.set_trace()
                init_vh_state = self.sim_env.get_vh_state(init_state)
                # print(colored(unsatisfied, "yellow"))
                self.particles[particle_id] = (
                    init_vh_state,
                    init_state,
                    satisfied,
                    unsatisfied,
                )
                # print(
                #     'init_state:',
                #     [
                #         edge
                #         for edge in init_state['edges']
                #         if 'HOLD' in edge['relation_type']
                #     ],
                # )

                self.particles_full[particle_id] = new_graph
            # print('-----')
            should_stop = False
            if self.agent_id == 2:
                # If agent 1 is grabbing an object, make sure that is not part of the plan
                new_goal_spec = copy.deepcopy(goal_spec)
                ids_grab_1 = [
                    edge["to_id"]
                    for edge in obs["edges"]
                    if edge["from_id"] == 1 and "hold" in edge["relation_type"].lower()
                ]
                if len(ids_grab_1) > 0:
                    for kgoal, elemgoal in new_goal_spec.items():
                        elemgoal["grab_obj_ids"] = [
                            ind
                            for ind in elemgoal["grab_obj_ids"]
                            if ind not in ids_grab_1
                        ]
                    should_stop = True
                goal_spec = new_goal_spec

            plan, root_node, subgoals = MCTS_utils.get_plan(
                self.mcts,
                self.particles,
                self.sim_env,
                nb_steps,
                goal_spec,
                last_plan,
                last_action,
                opponent_subgoal,
                length_plan=length_plan,
                verbose=self.verbose,
                num_process=self.num_processes,
            )

            if self.verbose:
                print("here")
                raise AssertionError

            # update last_loc, we will store the location of the objects we are trying to grab
            # at the moment of planning, if something changes, then we will replan when time comes
            elems_grab = []
            if subgoals is not None:
                for goal in subgoals:
                    if goal is not None and goal[0] is not None and "grab" in goal[0]:
                        elem_grab = int(goal[0].split("_")[1])
                        elems_grab.append(elem_grab)
            self.last_loc = {}
            for goal_id in elems_grab:
                self.last_loc[goal_id] = self.get_location_in_goal(obs, goal_id)

            # if len(plan) == 0 and self.agent_id == 1:
            #     ipdb.set_trace()

            if self.verbose:
                print(colored(plan[: min(len(plan), 10)], "cyan"))
            # ipdb.set_trace()
        # else:
        #     subgoals = [[None, None, None], [None, None, None]]
        # if len(plan) == 0 and not must_replan:
        #     ipdb.set_trace()
        #     print("Plan empty")
        #     raise Exception
        # print('-------- Plan {}: {}, {} ------------'.format(self.agent_id, lg, plan))
        if len(plan) > 0:
            action = plan[0]
            action = action.replace("[walk]", "[walktowards]")
        else:
            action = None
        if self.logging:
            info = {
                "plan": plan,
                "subgoals": subgoals,
                "belief": copy.deepcopy(self.belief.edge_belief),
                "belief_room": copy.deepcopy(self.belief.room_node),
            }

            if self.get_plan_states or self.get_plan_cost:
                plan_states = []
                plan_cost = []
                env = self.sim_env
                env.pomdp = True
                particle_id = 0
                vh_state = self.particles[particle_id][0]
                plan_states.append(vh_state.to_dict())
                for action_item in plan:
                    if self.get_plan_cost:
                        plan_cost.append(
                            env.compute_distance(vh_state, action_item, self.agent_id)
                        )

                    # if self.char_index == 1:
                    #     ipdb.set_trace()

                    success, vh_state = env.transition(
                        vh_state, {self.char_index: action_item}
                    )
                    vh_state_dict = vh_state.to_dict()
                    # print(action_item, [edge['to_id'] for edge in vh_state_dict['edges'] if edge['from_id'] == self.agent_id and edge['relation_type'] == 'INSIDE'])
                    plan_states.append(vh_state_dict)

                if self.get_plan_states:
                    info["plan_states"] = plan_states
                if self.get_plan_cost:
                    info["plan_cost"] = plan_cost

            if self.logging_graphs:
                info.update({"obs": obs["nodes"].copy()})
        else:
            info = {"plan": plan, "subgoals": subgoals}

        self.last_action = action
        self.last_subgoal = (
            subgoals if subgoals is not None and len(subgoals) > 0 else None
        )
        self.last_plan = plan
        # print(info['subgoals'])
        # print(action)
        time2 = time.time()
        # print("Time: ", time2 - time1)
        if self.verbose:
            print("Replanning... ", should_replan or must_replan)
        if should_replan:
            if self.verbose:
                print(
                    "Agent {} did replan: ".format(self.agent_id),
                    self.last_loc,
                    obj_grab,
                    curr_loc_index,
                    plan,
                )
            # if len(plan) == 0 and self.agent_id == 1:
            #     ipdb.set_trace()

        else:
            if self.verbose:
                print(
                    "Agent {} not replan: ".format(self.agent_id),
                    self.last_loc,
                    obj_grab,
                    curr_loc_index,
                    plan,
                )

        if action is not None and "grab" in action:
            if self.agent_id == 2:
                grab_id = int(action.split()[2][1:-1])
                grabbed_obj = [
                    edge
                    for edge in obs["edges"]
                    if edge["to_id"] == grab_id
                    and "hold" in edge["relation_type"].lower()
                ]
                if len(grabbed_obj):
                    raise AssertionError
            # if len([edge for edge in obs['edges'] if edge['from_id'] == 369 and edge['to_id'] == 103]) > 0:
            #     print("Bad plan")
            #     ipdb.set_trace()

        return action, info

    def reset(self, gt_graph):
        self.last_action = None
        self.last_subgoal = None
        self.failed_action = False
        self.init_gt_graph = gt_graph
        self.belief = belief.Belief(
            gt_graph,
            agent_id=self.agent_id,
            seed=self.seed,
            belief_params=self.belief_params,
        )
        self.sim_env.reset(gt_graph)
        add_bp = self.num_processes == 0
        self.mcts = MCTS.MCTS(
            gt_graph,
            self.agent_id,
            self.char_index,
            self.max_episode_length,
            self.num_simulation,
            self.max_rollout_steps,
            self.c_init,
            self.c_base,
            seed=self.seed,
            agent_params=self.agent_params,
            add_bp=add_bp,
        )

        # self.mcts.should_close = self.should_close
