import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.d_in = d_in
        raise NotImplementedError("Use your problem1 implementation")

    def init_weights(self):
        raise NotImplementedError("Use your problem1 implementation")

    def forward(self, input):
        raise NotImplementedError("Use your problem1 implementation")
