# ============================================================
# STDP: SPIKE-TIMING DEPENDENT PLASTICITY
# Biological learning rule: "neurons that fire together, wire
# together" (Hebb 1949; Bi & Poo 1998).
#   pre fires BEFORE post (dt > 0) -> potentiation (+A_plus * exp(-dt/tau_plus))
#   pre fires AFTER post  (dt < 0) -> depression   (-A_minus * exp(dt/tau_minus))
# ============================================================

import math

import torch


class STDPLearner:
    """Pairwise + online trace-based STDP on a (num_pre x num_post) weight matrix."""

    def __init__(self, num_pre, num_post, wmax=1.0, a_plus=0.02, a_minus=0.025,
                 tau_plus=20.0, tau_minus=20.0):
        self.w = torch.zeros(num_pre, num_post)
        self.wmax = wmax
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

    def delta(self, dt):
        """Weight change for one pre->post spike pair with time difference dt."""
        if dt > 0:
            return self.a_plus * math.exp(-dt / self.tau_plus)
        return -self.a_minus * math.exp(dt / self.tau_minus)

    def apply_pair(self, i, j, dt):
        """Classic pairwise STDP: fire pre at t, post at t+dt."""
        self.w[i, j] = (self.w[i, j] + self.delta(dt)).clamp(0.0, self.wmax)
        return self.w[i, j]

    # ---------- online trace version ----------
    def init_traces(self):
        self.pre_trace = torch.zeros(self.w.shape[0])
        self.post_trace = torch.zeros(self.w.shape[1])

    def step_traces(self, pre_spikes, post_spikes, decay=0.95):
        """Online STDP for one timestep. pre_spikes/post_spikes: 0/1 tensors."""
        if not hasattr(self, "pre_trace"):
            self.init_traces()
        pre = pre_spikes.float()
        post = post_spikes.float()
        # depression: post fired recently BEFORE this pre spike
        self.w = (self.w - self.a_minus * torch.outer(pre, self.post_trace)).clamp(0.0, self.wmax)
        # potentiation: pre fired recently BEFORE this post spike
        self.w = (self.w + self.a_plus * torch.outer(self.pre_trace, post)).clamp(0.0, self.wmax)
        self.pre_trace = decay * self.pre_trace + pre
        self.post_trace = decay * self.post_trace + post
        return self.w
