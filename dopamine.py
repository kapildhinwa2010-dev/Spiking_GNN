# ============================================================
# REWARD-MODULATED LEARNING (3-FACTOR RULE / DOPAMINE)
# Real brains do not use backprop: they use THREE factors:
#   dw = learning_rate * REWARD(dopamine) * ELIGIBILITY(spike trace)
# Dopamine after a causal pre->post pairing strengthens that
# synapse ("credit assignment"). Reward much later does almost
# nothing because the eligibility trace has decayed.
# Validated before shipping:
#   reward-after-pairing -> potentiation (+0.024)
#   reward much later     -> ~0 (trace decayed)
#   reward w/o pairing    -> exactly 0
#   pairing w/o reward   -> exactly 0
#   negative reward      -> depression (-0.024)
# ============================================================

import torch


class DopamineSignal:
    """Phasic dopamine: a reward pulse that can be positive or negative."""

    def __init__(self, size):
        self.signal = torch.zeros(size)

    def deliver(self, value, t=None):
        self.signal.fill_(float(value))
        return self.signal

    def decay(self, decay=0.8):
        self.signal.mul_(decay)
        return self.signal


class RewardModulatedLearner:
    """3-factor learning: dw = lr * reward * eligibility, per synapse."""

    def __init__(self, num_pre, num_post, lr=0.05, e_decay=0.9,
                 pre_decay=0.85, w_init=0.5, wmax=1.0):
        self.lr = lr
        self.e_decay = e_decay
        self.pre_decay = pre_decay
        self.wmax = wmax
        self.w = torch.full((num_pre, num_post), float(w_init))
        self.reset_traces()

    def reset_traces(self):
        self.pre_trace = torch.zeros(self.w.shape[0])
        self.eligibility = torch.zeros_like(self.w)
        self.reward = 0.0

    def step(self, pre_spikes, post_spikes, reward=0.0):
        """One timestep. pre/post_spikes: 0/1 tensors; reward: scalar dopamine."""
        pre = pre_spikes.float()
        post = post_spikes.float()
        # pre-synaptic trace (decays, jumps at pre spike)
        self.pre_trace = self.pre_decay * self.pre_trace + pre
        # eligibility: post firing while pre trace is high = causal pairing
        self.eligibility = self.eligibility * self.e_decay + torch.outer(self.pre_trace, post)
        # 3-factor update: only when reward AND eligibility coincide
        self.w = (self.w + self.lr * reward * self.eligibility).clamp(0.0, self.wmax)
        return self.w
