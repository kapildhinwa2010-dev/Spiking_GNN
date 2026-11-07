# ============================================================
# HOMEOSTATIC REGULATION (intrinsic plasticity)
# Real brains keep firing rates stable: if a neuron fires too
# much its threshold rises, too little and it drops. This module
# is the biological anti-seizure mechanism.
# ============================================================

import torch


class HomeostaticRegulator:
    """Adaptive threshold that drives firing rate toward a target."""

    def __init__(self, size, target_rate=0.2, lr=0.05, init_threshold=0.0):
        self.target_rate = target_rate
        self.lr = lr
        self.threshold = torch.full((size,), float(init_threshold))

    def step(self, rates):
        """rates: (N,) current firing rates. Returns updated thresholds."""
        self.threshold = self.threshold + self.lr * (rates - self.target_rate)
        return self.threshold

    def gate(self, membrane):
        """Threshold-subtracted membrane: use directly as spike drive."""
        return membrane - self.threshold
