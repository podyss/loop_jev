"""Training-only suffix decoding from single-pass student layer states.

An adaptation, not a reproduction of the paper's autoregressive latent curriculum.
The frozen decoder receives latent states and teacher-forced suffix tokens only.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def suffix_texts(steps, stages=3):
    # Keep at least one complete step even for short traces.
    return ["\n".join(steps[min(k * len(steps) // stages, max(0, len(steps) - 1)):])
            for k in range(stages)]


def pack_suffixes(sequences, latent_count, pad_id, device):
    width = latent_count + max(1, max(map(len, sequences))) - 1
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    mask[:, :latent_count] = 1
    for i, tokens in enumerate(sequences):
        n = len(tokens)
        if n:
            ids[i, latent_count:latent_count + n - 1] = torch.tensor(tokens[:-1], device=device)
            mask[i, :latent_count + n - 1] = 1
            labels[i, latent_count - 1:latent_count + n - 1] = torch.tensor(tokens, device=device)
    return ids, mask, labels


def suffix_prediction_loss(decoder, states, sequences, pad_id, embedding_scale=0.02):
    """Dense token CE, averaged per valid example; gradients reach all prefix states.

    The embedding hook executes inside the root model's FSDP forward. Checkpoint
    recomputation reinstalls it, so no temporary hook survives the call.
    """
    latent = torch.stack(states, dim=1)
    ids, mask, labels = pack_suffixes(sequences, len(states), pad_id, latent.device)
    keep = labels != -100
    if not keep.any():
        # Still run the decoder on every rank to preserve FSDP collective order.
        keep[0, len(states) - 1] = True
    targets = labels[keep]

    def decode(prefix):
        def inject(_module, _inputs, embeddings):
            normalized = F.normalize(prefix.float(), dim=-1) * (prefix.shape[-1] ** .5)
            replacement = (normalized * embedding_scale).to(embeddings.dtype)
            return torch.cat((replacement, embeddings[:, prefix.shape[1]:]), dim=1)
        handle = decoder.get_input_embeddings().register_forward_hook(inject)
        try:
            return decoder(input_ids=ids, attention_mask=mask, logits_to_keep=keep,
                           use_cache=False).logits
        finally:
            handle.remove()

    logits = checkpoint(decode, latent, use_reentrant=False)
    losses = F.cross_entropy(logits.float(), targets, ignore_index=-100, reduction="none")
    owners = keep.nonzero()[:, 0]
    sums = losses.new_zeros(len(sequences)).scatter_add(0, owners, losses)
    counts = (labels != -100).sum(dim=1)
    valid = counts > 0
    return (sums / counts.clamp_min(1) * valid).sum() / valid.sum().clamp_min(1)
