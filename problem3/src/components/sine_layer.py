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
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        if self.is_first:
            bound = 1.0 / self.d_in
        else:
            bound = (6.0 / self.d_in) ** 0.5 / self.omega_0
        self.linear.weight.uniform_(-bound, bound)
        if self.linear.bias is not None:
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
