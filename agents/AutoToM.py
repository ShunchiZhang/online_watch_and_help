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
        story_belief = list(belief.edge_belief.values())[0]
        self.story_ctnr_ids = set(story_belief["INSIDE"][0][1:])
        self.story_srfc_ids = set(story_belief["ON"][0][1:])
        self.init_story = EG(gt_graph).story(self.story_ctnr_ids, self.story_srfc_ids)

        self.human_init_room = self.saver.episode_saved_info["init_rooms"][0]

    def prepare(self, curr_gt_graph, human_actions):
        eg = EG(curr_gt_graph)

        # * curr_story
        curr_story = eg.story(
            self.story_ctnr_ids,
            self.story_srfc_ids,
        )

        # * curr_state: human close to ..., holds ...
        curr_state = eg.agent_state_natlang(agent_id=1, name="Human")

        # * actions
        actions = eg.actions_to_natlang(human_actions, self.human_init_room)

        return curr_story, curr_state, actions

    def step(self, curr_gt_graph, human_actions, particles):
        curr_story, curr_state, actions = self.prepare(curr_gt_graph, human_actions)
        match self.method:
            case "autotom":
                particles = self.particle_filter(
                    curr_story, curr_state, actions, particles
                )
            case "llm":
                particles = self.new_particles(actions, n=self.num_particles)
            case _:
                raise ValueError(f"Invalid method: {self.method}")
        return particles

    def new_particles(self, actions, n):
        # ! IMPORTANT: maintain overall goal as particles, instead of future goals.
        while True:
            try:
                resp, cost = prompts.call_gpt(
                    prompt="\n\n".join(
                        [
                            prompts.question,
                            prompts.propose(
                                story=self.init_story,
                                action="\n".join(actions),
                                n=n,
                            ),
                        ]
                    ),
                    llm_name=self.llm_name,
                )
                self.saver.debug(f"[new_particles] {cost}")

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

    def particle_filter(self, curr_story, curr_state, actions, particles):
        # ^ 1. fill
        if len(particles) < self.num_particles:
            new_particles = self.new_particles(actions, n=self.num_particles)
            particles.fill_particles(new_particles, self.num_particles)

        # ^ 2. reweight and normalize
        while True:
            try:
                probs = self.forward_likelihood(
                    curr_story, curr_state, actions[-1], particles
                )
                break
            except Exception as e:
                self.saver.error(f"[forward_likelihood] {e}")
                traceback.print_exc()
        self.saver.debug(f"[smc.fill]\n{pretty_repr(particles.to_natlang())}")
        particles.reweight(probs)
        self.saver.debug(f"[smc.reweight]\n{pretty_repr(particles.to_natlang())}")
        particles.filter_low_conf(self.filter_thres)
        self.saver.debug(f"[smc.filter]\n{pretty_repr(particles.to_natlang())}")
        return particles

    def forward_likelihood(self, curr_story, curr_state, curr_action, particles):
        probs = []
        # & >>>>> only for debug >>>>>
        saved_prompts = []
        saved_choices = []
        # & <<<<< only for debug <<<<<
        for particle in particles.particles:
            prompt = prompts.forward_likelihood(
                story=curr_story,
                state=curr_state,
                action=curr_action,
                particle=particle.to_natlang(),
            )
            resp, cost = prompts.call_gpt(prompt, self.llm_name)
            self.saver.debug(f"[forward_likelihood] {cost}")

            likelihood = prompts.Likelihood.model_validate(resp)
            probs.append(likelihood.likelihood)

            # & >>>>> only for debug >>>>>
            saved_prompts.append(prompt)
            saved_choices.append(particle.to_natlang())

        self.saver.debug(f"[forward] {curr_action}")
        self.saver.debug(f"[forward]\n{pretty_repr(dict(zip(saved_choices, probs)))}")
        # & <<<<< only for debug <<<<<

        partition = sum(probs)
        if partition == 0:
            probs = [1 / len(probs) for _ in probs]
        else:
            probs = [p / partition for p in probs]

        return probs
