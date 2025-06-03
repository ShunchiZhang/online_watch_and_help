import traceback

from pydantic import ValidationError

from agents import AutoToM_prompts as prompts


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
        self.particles = []

    def step(self, state, actions):
        match self.method:
            case "autotom":
                particles = self.particle_filter(state, actions)
            case "llm":
                particles = self.new_particles(actions, n=self.num_particles)
            case _:
                raise ValueError(f"Invalid method: {self.method}")
        return particles

    def new_particles(self, actions, n):
        # ! IMPORTANT: maintain overall goal as particles, instead of future goals.
        story = "\n\n".join([self.story, "\n".join(actions)])
        while True:
            try:
                resp, cost = prompts.call_gpt(
                    prompt="\n\n".join(
                        [
                            prompts.question,
                            prompts.propose(story=story, n=n),
                        ]
                    ),
                    llm_name=self.llm_name,
                )
                hypos = prompts.GoalParticles.model_validate(resp)
                hypos.normalize()
                assert len(hypos.particles) == n
                self.saver.debug(f"[new_particles] {cost}")
                return hypos
            except (AssertionError, ValidationError) as e:
                self.saver.error(f"{e}: {resp}")

    def particle_filter(self, state, actions):
        # ^ 1. fill
        assert len(actions) != 0
        if len(self.particles) == 0:
            particles = self.new_particles(actions, n=self.num_particles)
        else:
            # * filter
            particles = {
                h: p for h, p in self.particles.items() if p >= self.filter_thres
            }
            # * propose
            if len(particles) < self.num_particles:
                new_hypos = self.new_particles(actions, n=self.num_particles)
                for new_hypo, p in new_hypos.items():
                    if new_hypo not in particles:
                        particles[new_hypo] = p
                    if len(particles) == self.num_particles:
                        break
        # ^ 2. reweight and normalize
        while True:
            try:
                probs = self.forward_likelihood(state, actions[-1])
                l_probs, l_choices = len(probs), len(particles)
                assert l_probs == l_choices, f"{l_probs = } | {l_choices = }"
                # * normalize
                partition = sum(probs)
                if partition == 0:
                    probs = [1 / l_choices for _ in probs]
                else:
                    probs = [p / partition for p in probs]
                break
            except Exception as e:
                self.saver.error(f"[forward_likelihood] {e}")
                traceback.print_exc()
        self.saver.debug(f"{particles.keys()}")
        self.saver.debug(f"curr_p = {probs}")
        self.saver.debug(f"prev_p = {particles.values()}")
        partition = sum(p * p_old for p, p_old in zip(probs, particles.values()))
        particles = {
            h: p * p_old / partition for p, (h, p_old) in zip(probs, particles.items())
        }
        self.saver.debug(f"combined_p = {particles.values()}")
        particles = {
            h: p
            for i, (h, p) in enumerate(particles.items())
            # * remove the whole particle if current prob[i] is low
            if probs[i] >= self.filter_thres
        }
        self.saver.debug(f"filtered_p = {particles.values()}")
        partition = sum(particles.values())
        particles = {h: p / partition for h, p in particles.items()}
        return particles

    def forward_likelihood(self, state, action, particles):
        choices = list(particles.keys())
        probs = []

        story = "\n\n".join([self.story, state])

        for choice in choices:
            prompt2 = prompts.forward_likelihood(story, choice, action)
            resp, cost = prompts.call_gpt(prompt2, self.llm_name)
            self.saver.debug(f"[forward_likelihood] {cost}")
            likelihood = prompts.Likelihood.model_validate(resp)
            probs.append(likelihood.likelihood)
        return probs
