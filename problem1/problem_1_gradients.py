"""
For each function below
- y is the output of the model
- x is the input of the model

Use automatic differentiation to compute all functions of the gradient of y with respect to x.

Hint: implement the functions in order from top to bottom.
"""

import torch


def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor `y` that represents a function of `x` (`y` could be multi-dimensional),
    return the gradient of `y` with respect to `x`. It is important to note for the assignment
    that the PURPOSE of this function is to be used not only to compute gradients, but
    also to compute them in such a way that we can use them as part of the computational
    graph for backpropagation THROUGH the gradient function. If this is unclear, you
    may note that if you were to run the forward pass of a neural network, then
    compute the output of the backwards pass up to an input `x`, the concatenated
    [forwards pass, backwards pass] is a function of `x` and so defines a longer, 
    concatenated "forward pass" of `x`. We want to use this to train a SIREN on the gradients
    of an image, such that the SIREN's actual output corresponds to the image.
    
    Hint: You may find the `torch.autograd.grad` function useful. To achieve the stated purpose
    above, you will need to set a specific boolean parameter to one of two options. Can you
    read the documentation and figure out which one and why?
    """
    num_outputs = y.shape[-1]
    grads = []
    for channel_idx in range(num_outputs):
        # Coefficients for sum_ij (coeff_ij * y_ij); autograd.grad needs a scalar objective.
        y_coefficients = torch.zeros_like(y)
        y_coefficients[..., channel_idx] = 1.0
        grad_for_channel = torch.autograd.grad(
            y,
            x,
            grad_outputs=y_coefficients,
            create_graph=True,
            retain_graph=True,
        )[0]
        grads.append(grad_for_channel)
    if num_outputs == 1:
        return grads[0]
    return torch.stack(grads, dim=-2)


def divergence(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor `y` that represents a function of `x` (`y` could be multi-dimensional),
    return the divergence of `y` with respect to `x`. By convention we will compute
    the divergence along the LAST axis of `y`. When we use this function it will usually
    be flattened in practice.

    Hint: You may find the ` torch.autograd.grad` function useful. Like in `gradient`,
    you need to set a specific boolean parameter to the correct out of two options.
    """
    jac = gradient(y, x)
    n = min(y.shape[-1], x.shape[-1])
    idx = torch.arange(n, device=jac.device)
    if jac.dim() == 2:
        return jac[..., idx].sum(dim=-1, keepdim=True)
    return jac[..., idx, idx].sum(dim=-1, keepdim=True)


def laplace(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor `y` that represents a function of `x` (`y` could be multi-dimensional),
    return the laplacian of `y` with respect to `x`.
    
    Hint: You may find some of our previous functions useful and the identity in the
    `Gradient` section of Wikipedia: https://en.wikipedia.org/wiki/Laplace_operator#Generalization.
    """
    grad_y = gradient(y, x)
    if grad_y.dim() == 2:
        return divergence(grad_y, x)
    num_channels = grad_y.shape[-2]
    laps = [divergence(grad_y[..., c, :], x) for c in range(num_channels)]
    return torch.cat(laps, dim=-1)
