import copy

from rich.pretty import pretty_repr

from agents import AutoToM_prompts as prompts
from utils.utils_exception import ParticleLengthException, handle
from utils.utils_graph import EG, check_progress, item


class AutoToM:
    def __init__(
        self,
        filter_thres,
        num_particles,
        proposer_name,
        estimator_name,
        method,
        hide_helper_history,
        disable_estimation,
    ):
        """
        `filter_thres`: filter threshold to filter out low-confidence hypos
        `num_particles`: number of particles to use
        `llm_name`: name of the llm to use, e.g., "gpt-4o" or "o3-mini"
        `method`: method to use, e.g., "autotom" or "llm"
        """
        self.proposer_name = proposer_name
        self.estimator_name = estimator_name
        self.filter_thres = filter_thres
        self.num_particles = num_particles
        self.method = method
        self.hide_helper_history = hide_helper_history
        self.disable_estimation = disable_estimation

    def reset(self, gt_graph, belief):
        """
        prepare
        - init_story (for self.new_particles())
        - curr_story (for self.forward_likelihood())
        """
        env_belief = list(belief.edge_belief.values())[0]
        self.env_ctnr_ids = set(env_belief["INSIDE"][0][1:])
        self.env_srfc_ids = set(env_belief["ON"][0][1:])
        # self.init_env_state = EG(gt_graph).env_state(
        #     self.env_ctnr_ids, self.env_srfc_ids
        # )

    def prepare(self, init_gt_graph, curr_gt_graph, human_actions, helper_actions):
        eg_init = EG(init_gt_graph)
        eg_curr = EG(curr_gt_graph)

        # * curr_story
        init_env_state = eg_init.env_state(self.env_ctnr_ids, self.env_srfc_ids)
        curr_env_state = eg_curr.env_state(self.env_ctnr_ids, self.env_srfc_ids)

        # * curr_state: human close to ..., holds ...
        init_human_state = eg_init.agent_state_natlang(agent_id=1, name="Human")
        curr_human_state = eg_curr.agent_state_natlang(agent_id=1, name="Human")

        # * actions
        human_done, _, _ = check_progress(human_actions[:-1])
        human_actions = eg_curr.actions_to_natlang(human_actions, name="Human")
        helper_actions = eg_curr.actions_to_natlang(helper_actions, name="Helper")

        def filter_key_action(a):
            return any(verb in a for verb in ["grabs", "puts"])

        human_key_history = list(filter(filter_key_action, human_actions[:-1]))
        if len(human_key_history) > 0:
            human_key_history = "\n".join(human_key_history)
        else:
            human_key_history = "Human has not taken any key action yet."

        helper_key_history = list(filter(filter_key_action, helper_actions[:-1]))
        if len(helper_key_history) > 0:
            helper_key_history = "\n".join(helper_key_history)
        else:
            helper_key_history = "Helper has not taken any key action yet."

        if self.hide_helper_history:
            key_action_history = human_key_history
        else:
            key_action_history = "\n\n".join([human_key_history, helper_key_history])

        prompt_info = dict(
            init_env_state=init_env_state,
            curr_env_state=curr_env_state,
            init_human_state=init_human_state,
            curr_human_state=curr_human_state,
            key_action_history=key_action_history,
            next_human_action=human_actions[-1],
        )
        self.saver.record_prepare(prompt_info, human_done)
        return prompt_info, human_done

    def step(
        self, init_gt_graph, curr_gt_graph, human_actions, helper_actions, particles
    ):
        prompt_info, human_done = self.prepare(
            init_gt_graph, curr_gt_graph, human_actions, helper_actions
        )
        match self.method:
            case "autotom":
                particles = self.particle_filter(prompt_info, particles, human_done)
            case "llm":
                particles = self.new_particles(prompt_info, n=self.num_particles)
            case _:
                raise ValueError(f"Invalid method: {self.method}")
        return particles

    def new_particles(self, prompt_info, n):
        if n == 1:
            prompt = prompts.propose_single(**prompt_info)
            out_type = prompts.GoalParticle
        else:
            prompt = prompts.propose(**prompt_info, n=n)
            out_type = prompts.GoalParticles
        self.saver.debug(f"[new_particles.prompt]\n{prompt}")

        while True:
            try:
                particles, io, cost = prompts.call_llm_batch(
                    [prompt],
                    self.proposer_name,
                    out_type=out_type,
                    # temperature=1.0,
                )
                self.saver.record_io(io)
                self.saver.record_cost(cost, "new_particles")
                particles = item(particles)

                if len(particles) != n:
                    raise ParticleLengthException(f"{len(particles) = } != {n}")
                break

            except Exception as e:
                handle(e, self.saver)

        particles.normalize()
        self.saver.info(f"[new_particles]\n{pretty_repr(particles.to_natlang())}")

        self.saver.warning(f"[llm.probs] {particles.probs_grab(in_log=True)}")
        self.saver.warning(f"[llm.probs] {particles.probs_put(in_log=True)}")

        return particles

    def particle_filter(self, prompt_info, particles, human_done):
        # ^ 1. fill
        if len(particles) < self.num_particles:
            new_particles = self.new_particles(prompt_info, n=self.num_particles)
            particles.fill_particles(new_particles, self.num_particles)
        self.saver.info(f"[smc.fill]\n{pretty_repr(particles.to_natlang())}")

        if not self.disable_estimation:
            # ^ 2. estimate and reweight
            probs = self.forward_likelihood(prompt_info, particles, human_done)
            particles.reweight(probs)
            self.saver.info(f"[smc.reweight]\n{pretty_repr(particles.to_natlang())}")

        # ^ 3. filter
        particles.filter_low_conf(self.filter_thres, min_num=5, normalize=False)
        self.saver.info(f"[smc.filter]\n{pretty_repr(particles.to_natlang())}")

        self.saver.warning(f"[autotom.probs] {particles.probs_grab(in_log=True)}")
        self.saver.warning(f"[autotom.probs] {particles.probs_put(in_log=True)}")
        return particles

    def forward_likelihood(self, prompt_info, particles, human_done):
        if "all_time":
            log_prompt = prompts.forward_likelihood_all_time(
                **prompt_info, goal="DEBUG"
            )
            self.saver.debug(f"[forward.prompt]\n{log_prompt}")

            forward_particles = copy.deepcopy(particles)
        else:
            log_prompt = prompts.forward_likelihood(**prompt_info, goal="DEBUG")
            self.saver.debug(f"[forward.prompt]\n{log_prompt}")

            forward_particles = copy.deepcopy(particles)
            forward_particles.minus_objects(human_done)

        goals = [particle.to_natlang() for particle in forward_particles.particles]
        batch = [prompts.forward_likelihood(**prompt_info, goal=goal) for goal in goals]
        probs, _, cost = prompts.call_llm_batch(
            batch, self.estimator_name, out_type="forward_likelihood"
        )
        self.saver.record_cost(cost, "forward_likelihood")
        # probs = [p.likelihood for p in probs]

        self.saver.info(f"[forward]\n{pretty_repr(dict(zip(goals, probs)))}")

        for ith_particle, particle in enumerate(forward_particles.particles):
            if len(particle.objects) == 0:
                probs[ith_particle] = 0

        if sum(probs) == 0:
            probs = [1e-2 for _ in probs]

        return probs
