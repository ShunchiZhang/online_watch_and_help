import traceback

from pydantic import ValidationError
from rich.pretty import pretty_repr

from agents import AutoToM_prompts as prompts
from utils.utils_graph import EG


class AutoToM:
    def __init__(
        self,
        filter_thres,
        num_particles,
        llm_name,
        method,
    ):
        """
        `filter_thres`: filter threshold to filter out low-confidence hypos
        `num_particles`: number of particles to use
        `llm_name`: name of the llm to use, e.g., "gpt-4o" or "o3-mini"
        `method`: method to use, e.g., "autotom" or "llm"
        """
        self.llm_name = llm_name
        self.filter_thres = filter_thres
        self.num_particles = num_particles
        self.method = method

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

    def prepare(self, curr_gt_graph, human_actions, helper_actions):
        eg = EG(curr_gt_graph)

        # * curr_story
        curr_env_state = eg.env_state(self.env_ctnr_ids, self.env_srfc_ids)

        # * curr_state: human close to ..., holds ...
        curr_human_state = eg.agent_state_natlang(agent_id=1, name="Human")

        # * actions
        human_actions = eg.actions_to_natlang(human_actions, name="Human")
        helper_actions = eg.actions_to_natlang(helper_actions, name="Helper")

        def filter_key_action(a):
            return any(verb in a for verb in ["grabs", "puts"])

        human_key_history = filter(filter_key_action, human_actions[:-1])
        if len(list(human_key_history)) > 0:
            human_key_history = "\n".join(human_key_history)
        else:
            human_key_history = "Human has not taken any key action yet."

        helper_key_history = filter(filter_key_action, helper_actions[:-1])
        if len(list(helper_key_history)) > 0:
            helper_key_history = "\n".join(helper_key_history)
        else:
            helper_key_history = "Helper has not taken any key action yet."

        key_action_history = "\n\n".join(human_key_history + helper_key_history)

        # return curr_env_state, curr_human_state, key_history, human_actions[-1]
        return dict(
            curr_env_state=curr_env_state,
            curr_human_state=curr_human_state,
            key_action_history=key_action_history,
            next_human_action=human_actions[-1],
        )

    def step(self, curr_gt_graph, human_actions, helper_actions, particles):
        prompt_info = self.prepare(curr_gt_graph, human_actions, helper_actions)
        match self.method:
            case "autotom":
                particles = self.particle_filter(prompt_info, particles)
            case "llm":
                particles = self.new_particles(prompt_info, n=self.num_particles)
            case _:
                raise ValueError(f"Invalid method: {self.method}")
        return particles

    def new_particles(self, prompt_info, n):
        while True:
            try:
                prompt = prompts.propose(**prompt_info, n=n)
                resp, cost = prompts.call_gpt(prompt, self.llm_name)
                self.saver.record_cost(cost, "new_particles")

                particles = prompts.GoalParticles.model_validate(resp)

                assert len(particles) == n
                break

            except (AssertionError, ValidationError) as e:
                self.saver.error(f"{e}: {resp}")
            except Exception as e:
                self.saver.error(f"{e}: {resp}")
                traceback.print_exc()

        particles.normalize()
        return particles

    def particle_filter(self, prompt_info, particles):
        # ^ 1. fill
        if len(particles) < self.num_particles:
            new_particles = self.new_particles(prompt_info, n=self.num_particles)
            particles.fill_particles(new_particles, self.num_particles, normalize=False)

        # ^ 2. reweight and normalize
        while True:
            try:
                probs = self.forward_likelihood(prompt_info, particles)
                break
            except Exception as e:
                self.saver.error(f"[forward_likelihood] {e}")
                traceback.print_exc()
        self.saver.debug(f"[smc.fill]\n{pretty_repr(particles.to_natlang())}")
        particles.reweight(probs)
        self.saver.debug(f"[smc.reweight]\n{pretty_repr(particles.to_natlang())}")
        particles.filter_low_conf(self.filter_thres, normalize=False)
        self.saver.debug(f"[smc.filter]\n{pretty_repr(particles.to_natlang())}")
        return particles

    def forward_likelihood(self, prompt_info, particles):
        probs = []
        goals = [particle.to_natlang() for particle in particles.particles]
        for goal in goals:
            prompt = prompts.forward_likelihood(**prompt_info, goal=goal)
            resp, cost = prompts.call_gpt(prompt, self.llm_name)
            self.saver.record_cost(cost, "forward_likelihood")

            likelihood = prompts.Likelihood.model_validate(resp)
            probs.append(likelihood.likelihood)

        # & >>>>> only for debug >>>>>
        log_prompt = prompts.forward_likelihood(**prompt_info, goal="[debug]")
        self.saver.debug(f"[forward]\n{log_prompt}")
        self.saver.debug(f"[forward]\n{pretty_repr(dict(zip(goals, probs)))}")
        # & <<<<< only for debug <<<<<

        partition = sum(probs)
        if partition == 0:
            probs = [1 / len(probs) for _ in probs]
        else:
            probs = [p / partition for p in probs]

        return probs
