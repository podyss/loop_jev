"""Finite-horizon extrapolation of damped block increments in a local subspace."""

import torch


def extrapolate(path, mask, *, steps, method):
    if len(path) != 4 or steps < 1 or method not in ("linear", "dmd"):
        raise ValueError("Extrapolation requires three observed updates and a positive horizon")
    end = path[-1].float()
    last = end - path[-2].float()
    if method == "linear":
        return (end + steps * last).to(path[-1].dtype)
    first = path[1].float() - path[0].float()
    middle = path[2].float() - path[1].float()
    basis = torch.stack((first, middle), dim=1).flatten(2)
    targets = torch.stack((middle, last), dim=1).flatten(2)
    if mask is not None:
        weights = mask[..., None].expand_as(last).flatten(1).unsqueeze(1)
        basis, targets = basis * weights, targets * weights
    gram = basis @ basis.transpose(1, 2)
    cross = basis @ targets.transpose(1, 2)
    start = cross[..., 1:2]
    ridge = 0.01 * (gram[:, 0, 0] + gram[:, 1, 1]).clamp_min(1e-6)
    a, b, d = gram[:, 0, 0] + ridge, gram[:, 0, 1], gram[:, 1, 1] + ridge
    inverse = torch.stack((d, -b, -b, a), dim=-1).reshape(-1, 2, 2)
    inverse = inverse / (a * d - b.square())[:, None, None]
    # Propagate residual increments in coefficient space, keeping the last observed
    # component outside the subspace as a constant rather than discarding it.
    projection = inverse @ start
    advance = inverse @ cross
    current = projection
    total = torch.zeros_like(current)
    for _ in range(steps):
        current = advance @ current
        total = total + current
    correction = ((total - steps * projection).transpose(1, 2) @ basis).reshape_as(last)
    baseline = steps * last
    # Guard against explosive extrapolation on a short or ill-conditioned trajectory.
    norm = correction.flatten(1).norm(dim=1).clamp_min(1e-6)
    limit = 0.75 * (baseline if mask is None else baseline * mask[..., None]).flatten(1).norm(dim=1)
    update = baseline + correction * (limit / norm).clamp(max=1)[:, None, None]
    return (end + update).to(path[-1].dtype)


def segmented_extrapolate(path, mask, *, steps, workspace_width):
    """Allow the question and the latent/answer tail to have different evolution."""
    split = path[-1].shape[1] - workspace_width
    if split <= 0:
        raise ValueError("Workspace width must leave a question prefix")
    prefix, tail = (
        extrapolate(tuple(state[:, region] for state in path), None if mask is None else mask[:, region],
                    steps=steps, method="dmd")
        for region in (slice(None, split), slice(split, None))
    )
    return torch.cat((prefix, tail), dim=1)


def parallel_finish(path, evaluate, *, rounds, step, nodes, sweeps):
    """Predict a path, then integrate real block increments at parallel time nodes."""
    if len(path) != 3 or path[-1].shape[0] != 1 or nodes < 1 or rounds % nodes or sweeps not in (1, 2):
        raise ValueError("Parallel finish needs two serial rounds and evenly divided time nodes")
    final = path[-1]
    base = final.float()
    stride = rounds // nodes
    starts = torch.arange(nodes, device=base.device, dtype=torch.float32) * stride
    # Midpoints of groups of discrete left-endpoint Euler updates.
    times = starts + (stride - 1) / 2
    predicted = base + times[:, None, None] * (base - path[-2].float())
    for sweep in range(sweeps):
        states = predicted.to(final.dtype)
        increments = step * (evaluate(states).float() - states.float())
        if sweep + 1 < sweeps:
            weights = (times[:, None] - starts[None, :]).clamp(0, stride)
            predicted = base + (weights @ increments.flatten(1)).reshape_as(predicted)
    return (base + stride * increments.sum(dim=0, keepdim=True)).to(final.dtype)
