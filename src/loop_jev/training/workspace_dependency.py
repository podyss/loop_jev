"""Train a shared block to consume a previous round's numeric workspace."""

import operator
import re
from fractions import Fraction

import torch
from torch import nn
from torch.nn import functional as F

from loop_jev.training.workspace_query import ALPHABET, EOS, IGNORE, NUMBER_TOKENS


_NUMBER = r"[+-]?\d+(?:\.\d+)?"
_BINARY = re.compile(rf"({_NUMBER})([+*/-])({_NUMBER})")
_OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


def dependency(rationale):
    """Find one adjacent, numerically checked edge in calculator annotations."""
    preceding = None
    seen = set()
    duplicates = set()
    for expression, label in re.findall(r"<<([^<>=]+)=([^<>=]+)>>", rationale):
        expression, label = (re.sub(r"[\s,$]", "", text) for text in (expression, label))
        match = _BINARY.fullmatch(expression)
        if not match or not re.fullmatch(_NUMBER, label):
            preceding = None
            if re.fullmatch(_NUMBER, label):
                result = Fraction(label)
                if result in seen:
                    duplicates.add(result)
                seen.add(result)
            continue
        left, op, right = match.groups()
        a, b, result = map(Fraction, (left, right, label))
        calculate = _OPS[op]
        if (op == "/" and not b) or calculate(a, b) != result:
            preceding = None
            continue
        if preceding is not None:
            source, source_label, operands = preceding
            if (source not in duplicates and source not in operands and (a == source) != (b == source)
                    and result != source and (op != "*" or (a and b))
                    and (op != "/" or a)):
                masked = f"x {op} {right}" if a == source else f"{left} {op} x"
                return masked, source_label, label.lstrip("+")
        if result in seen:
            duplicates.add(result)
        seen.add(result)
        preceding = result, label.lstrip("+"), (a, b)
    return None


def encode_dependency(tokenizer, rationale):
    edge = dependency(rationale)
    if edge is not None:
        masked, *numbers = edge
        if all(len(number) < NUMBER_TOKENS for number in numbers):
            ids = tokenizer.encode(f"\nCompute {masked} =", add_special_tokens=False)
            targets = []
            for number in numbers:
                digits = [ALPHABET.index(ch) for ch in number] + [EOS]
                targets.append(digits + [IGNORE] * (NUMBER_TOKENS - len(digits)))
            return ids, targets
    return tokenizer.encode("\nCompute x =", add_special_tokens=False), [[IGNORE] * NUMBER_TOKENS] * 2


class DependencyReadout(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Sequential(nn.RMSNorm(hidden_size), nn.Linear(hidden_size, NUMBER_TOKENS * (EOS + 1)))
            for _ in range(2)
        )

    def forward(self, writer, reader, masked_reader):
        logits = torch.stack((self.heads[0](writer.float()), self.heads[1](reader.float())), dim=1)
        masked = None
        if masked_reader is not None:
            with torch.no_grad():
                masked = self.heads[1](masked_reader.float()).unflatten(-1, (NUMBER_TOKENS, EOS + 1))
        return logits.unflatten(-1, (NUMBER_TOKENS, EOS + 1)), masked


def dependency_loss(logits, masked, targets):
    valid = targets.ne(IGNORE)
    loss = F.cross_entropy(logits.flatten(0, -2).float(), targets.flatten(),
                           ignore_index=IGNORE, reduction="sum") / valid.sum().clamp_min(1)
    with torch.no_grad():
        covered = valid.any(dim=-1)
        exact = (logits.argmax(-1).eq(targets) | ~valid).all(dim=-1) & covered
        stats = torch.zeros(6, dtype=torch.long, device=targets.device)
        stats[:2] = exact.sum(dim=0)
        stats[2] = covered[:, 1].sum()
        if masked is not None:
            stats[3] = exact[:, 1].sum()
            stats[4] = ((masked.argmax(-1).eq(targets[:, 1]) | ~valid[:, 1]).all(-1)
                        & covered[:, 1]).sum()
            stats[5] = stats[2]
    return loss, stats
