"""Weight-only Marlin projections for the static-graph benchmark."""

import torch
from torch import nn


def pack_groups(weight, bits, group_size):
    """Return packed GPTQ-layout codes and their unpermuted scales."""
    n, k = weight.shape
    if bits not in (4, 8) or k % group_size or n % 64:
        raise ValueError('Unsupported Marlin weight shape or bit width')
    group = weight.float().reshape(n, k // group_size, group_size)
    bias = 1 << (bits - 1)
    scale = (group.abs().amax(-1) / (bias - 1)).clamp_min(1e-8)
    codes = ((group / scale[..., None]).round().clamp(-bias, bias - 1).int() + bias).reshape(n, k)
    factor = 32 // bits
    packed = codes[:, ::factor].clone()
    for i in range(1, factor):
        packed |= codes[:, i::factor] << (bits * i)
    return packed.t().contiguous(), scale.t().contiguous().to(weight.dtype)


class PackedLinear(nn.Module):
    def __init__(self, linear, bits, workspace, *, group_size=128):
        super().__init__()
        # Load the extension without vLLM's engine or NVML discovery.
        from vllm import _C  # noqa: F401
        from vllm.scalar_type import scalar_types

        if linear.bias is not None:
            raise ValueError('Only bias-free projections are supported')
        self.k, self.n = linear.in_features, linear.out_features
        self.weight_type = (scalar_types.uint4b8 if bits == 4 else scalar_types.uint8b128).id
        packed, scale = pack_groups(linear.weight, bits, group_size)
        empty = torch.empty(0, dtype=torch.int32, device=packed.device)
        self.register_buffer('weight', torch.ops._C.gptq_marlin_repack(packed, empty, self.k, self.n, bits, False))
        # Marlin's grouped scale tile is stored transposed in each group of 64.
        self.register_buffer('scale', scale.reshape(-1, 8, 8).transpose(1, 2).reshape_as(scale).contiguous())
        self.register_buffer('empty', empty)
        self.register_buffer('workspace', workspace)

    def forward(self, x):
        value = x.reshape(-1, self.k)
        result = torch.ops._C.marlin_gemm(value, None, self.weight, None, self.scale, None, None, None,
                                         self.empty, self.empty, self.workspace, self.weight_type,
                                         value.size(0), self.n, self.k, True, False, True, False)
        return result.reshape(*x.shape[:-1], self.n)


def pack_backbone(base, bits, scope):
    device = base.readout.device
    workspace = torch.zeros(4 * torch.cuda.get_device_properties(device).multi_processor_count,
                            device=device, dtype=torch.int32)
    for i, layer in enumerate(base.model.layers):
        if scope == 'shared' and not 9 <= i < 18:
            continue
        for group in (layer.self_attn, layer.mlp):
            for name, proj in list(group.named_children()):
                if isinstance(proj, nn.Linear):
                    setattr(group, name, PackedLinear(proj, bits, workspace))
