class ExperienceBuffer:
    def __init__(
        self,
        observations=None,
        actions=None,
        rewards=None,
        dones=None,
        log_probs=None,
        learnable=None,
        infos=None,
    ):
        self.result = 0
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.learnable = []
        self.infos = []
        self.total_learnable = 0

        if observations is not None:
            self.observations = observations

        if actions is not None:
            self.actions = actions

        if rewards is not None:
            self.rewards = rewards

        if dones is not None:
            self.dones = dones  # TODO Done probably doesn't need to be a list, will always just be false until last?

        if log_probs is not None:
            self.log_probs = log_probs

        if learnable is not None:
            self.learnable = learnable
            self.total_learnable = sum(learnable)

        if infos is not None:
            self.infos = infos

    def size(self):
        return len(self.rewards)

    def learnable_size(self):
        return self.total_learnable

    def add_step(self, observation, action, reward, done, log_prob, learnable, info):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.learnable.append(learnable)
        self.total_learnable += learnable
        self.infos.append(info)

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.learnable = []
        self.infos = []

    def generate_slices(self, batch_size):
        for i in range(0, len(self.observations), batch_size):
            yield ExperienceBuffer(
                self.observations[i : i + batch_size],
                self.actions[i : i + batch_size],
                self.rewards[i : i + batch_size],
                self.dones[i : i + batch_size],
                self.log_probs[i : i + batch_size],
                self.learnable[i : i + batch_size],
                self.infos[i : i + batch_size],
            )
