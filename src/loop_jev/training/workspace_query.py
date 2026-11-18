"""Training-only numeric readout of intermediate GSM reasoning quantities."""

import re

import torch
from torch import nn
from torch.nn import functional as F


QUERY_COUNT = 3
QUERY_TOKENS = 16
NUMBER_TOKENS = 13
QUERY_VOCAB = 8192
ALPHABET = "0123456789.-/"
EOS = len(ALPHABET)
IGNORE = -100
_ANNOTATION = re.compile(r"<<[^<>]*?=\s*([^<>]+?)\s*>>")
_NUMBER = re.compile(r"[+-]?\d[\d,]*(?:\.\d+|/\d+)?")
_CLAUSE_END = re.compile(r"\n|[!?;]|\.(?=\s|$)")


def intermediate_queries(rationale):
    matches = list(_ANNOTATION.finditer(rationale))
    pairs = []
    for match, following in zip(matches, matches[1:]):
        number = match[1].replace(",", "").lstrip("+")
        if not re.fullmatch(r"-?\d+(?:[./]\d+)?", number) or len(number) >= NUMBER_TOKENS:
            continue
        tail = rationale[match.end():following.start()].lstrip(" $£€")
        repeated = _NUMBER.match(tail)
        if repeated:
            tail = tail[repeated.end():]
        phrase = _CLAUSE_END.split(tail, maxsplit=1)[0]
        # Drop formulas and numeric hints, retaining the quantity's description.
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", phrase.lower())
        words = [word for word in words if word not in {"x", "plus", "minus"}]
        if words:
            pairs.append((" ".join(words[:12]), number))
    # A duplicate description cannot select between two intermediate values.
    counts = {}
    for query, number in pairs:
        counts.setdefault(query, set()).add(number)
    unique = [(query, next(iter(values))) for query, values in counts.items() if len(values) == 1]
    if len(unique) <= QUERY_COUNT:
        return unique
    return [unique[i * (len(unique) - 1) // (QUERY_COUNT - 1)] for i in range(QUERY_COUNT)]


def encode_queries(tokenizer, rationale):
    queries, targets = [], []
    for query, number in intermediate_queries(rationale):
        ids = tokenizer.encode(query, add_special_tokens=False)[:QUERY_TOKENS]
        queries.append([1 + i % (QUERY_VOCAB - 1) for i in ids] + [0] * (QUERY_TOKENS - len(ids)))
        values = [ALPHABET.index(ch) for ch in number] + [EOS]
        targets.append(values + [IGNORE] * (NUMBER_TOKENS - len(values)))
    while len(queries) < QUERY_COUNT:
        queries.append([0] * QUERY_TOKENS)
        targets.append([IGNORE] * NUMBER_TOKENS)
    return queries, targets


class WorkspaceQueryReadout(nn.Module):
    def __init__(self, hidden_size, width=96):
        super().__init__()
        self.query_embedding = nn.Embedding(QUERY_VOCAB, width, padding_idx=0)
        self.query_position = nn.Parameter(torch.randn(QUERY_TOKENS, width) * 0.02)
        self.query_map = nn.Sequential(nn.Linear(width, width), nn.GELU())
        self.state_map = nn.Sequential(nn.RMSNorm(hidden_size), nn.Linear(hidden_size, width))
        self.decoder = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(),
                                     nn.Linear(width, NUMBER_TOKENS * (EOS + 1)))

    def forward(self, workspace, query_ids):
        # workspace: [rounds, batch, slots, hidden]; query_ids: [batch, queries, tokens]
        mask = query_ids.ne(0).unsqueeze(-1)
        embeddings = self.query_embedding(query_ids) * (1 + self.query_position)
        query = self.query_map((embeddings * mask).sum(dim=-2) / mask.sum(dim=-2).clamp_min(1))
        state = self.state_map(workspace)
        attention = torch.einsum("bqd,rbsd->rbqs", query, state) * (state.size(-1) ** -0.5)
        context = torch.einsum("rbqs,rbsd->rbqd", attention.softmax(dim=-1), state)
        query = query.unsqueeze(0).expand(state.shape[0], -1, -1, -1)
        logits = self.decoder(torch.cat((query, context), dim=-1))
        with torch.no_grad():
            masked = self.decoder(torch.cat((query[0], torch.zeros_like(context[0])), dim=-1))
        return logits.unflatten(-1, (NUMBER_TOKENS, EOS + 1)), masked.unflatten(-1, (NUMBER_TOKENS, EOS + 1))


def query_loss(logits, masked_logits, targets):
    valid = targets.ne(IGNORE)
    labels = targets.unsqueeze(0).expand(logits.shape[0], -1, -1, -1)
    total = F.cross_entropy(logits.reshape(-1, EOS + 1).float(), labels.reshape(-1),
                            ignore_index=IGNORE, reduction="sum")
    loss = total / (logits.shape[0] * valid.sum().clamp_min(1))
    with torch.no_grad():
        valid_query = valid.any(dim=-1)
        correct = ((logits.argmax(-1).eq(labels) | ~valid).all(-1) & valid_query).sum(dim=(1, 2))
        masked_correct = ((masked_logits.argmax(-1).eq(targets) | ~valid).all(-1) & valid_query).sum()
        stats = torch.cat((correct, masked_correct.reshape(1), valid_query.sum().reshape(1)))
    return loss, stats
