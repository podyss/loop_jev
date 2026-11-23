"""Compile checked calculator traces into two composable unary affine functions."""

import ast
import math
import operator
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

import torch
from torch import nn
from torch.nn import functional as F


NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
ANNOTATION = re.compile(r"<<([^<>=]+)=([^<>=]+)>>")
OPERATORS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
CALCULATE = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
MAX_COEFFICIENT = 1_000_000


@dataclass(frozen=True)
class Program:
    functions: tuple
    signature: str

    def targets(self):
        return [[math.asinh(value) for value in pair] for pair in self.functions]


def _expression(text, refs, inputs, literals):
    text = re.sub(r"[\s,$]", "", text)
    if len(text) > 120 or not re.fullmatch(r"[0-9.+*/()-]+", text):
        raise ValueError("Unsupported calculator expression")

    def convert(node):
        if isinstance(node, ast.Constant) or (
            isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant)
        ):
            value = Fraction(ast.get_source_segment(text, node))
            indices = refs.get(value, [])
            if indices:
                if value in inputs or len(indices) != 1:
                    # Ambiguity matters only if this branch reaches the answer.
                    return ("ambiguous", value)
                return ("ref", indices[0])
            literals[value] += 1
            return ("literal", value)
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return (OPERATORS[type(node.op)], convert(node.left), convert(node.right))
        raise ValueError("Unsupported calculator node")

    return convert(ast.parse(text, mode="eval").body)


def _affine(steps, root, *, literal=None, pivot=None):
    """Return exact (slope, intercept) and the operator tree from variable to output."""
    cache = {}

    def visit(node):
        op, value, *tail = node
        if op == "ref":
            if value == pivot:
                return 1, 0, "x"
            if value not in cache:
                cache[value] = visit(steps[value])
            return cache[value]
        if op == "ambiguous":
            raise ValueError("Ambiguous intermediate reference")
        if op == "literal":
            return (1, 0, "x") if value == literal else (0, value, "c")
        a, b, left = visit(value)
        c, d, right = visit(tail[0])
        if op in ("+", "-"):
            fn = CALCULATE[op]
            slope, intercept = fn(a, c), fn(b, d)
        elif op == "*" and not (a and c):
            slope, intercept = a * d + c * b, b * d
        elif op == "/" and not c and d:
            slope, intercept = Fraction(a) / d, Fraction(b) / d
        else:
            raise ValueError("Not an affine expression")
        return slope, intercept, f"({left}{op}{right})" if slope else "c"

    return visit(("ref", root))


def _verified_trace(row):
    annotations = ANNOTATION.findall(row.get("rationale", ""))
    if not 2 <= len(annotations) <= 16:
        return None
    try:
        inputs = Counter(Fraction(token.replace(",", "")) for token in NUMBER.findall(row["question"]))
        gold = Fraction(str(row["options"][row["gold_index"]]).replace(",", ""))
        steps, values, refs, literals = [], [], {}, Counter()
        for text, label in annotations:
            value = Fraction(re.sub(r"[\s,$]", "", label))
            expr = _expression(text, refs, inputs, literals)

            def factual(node):
                op, operand, *tail = node
                if op in ("literal", "ambiguous"):
                    return operand
                if op == "ref":
                    return values[operand]
                return CALCULATE[op](factual(operand), factual(tail[0]))

            if factual(expr) != value:
                return None
            refs.setdefault(value, []).append(len(steps))
            steps.append(expr)
            values.append(value)
        if values[-1] != gold:
            return None
    except (ValueError, SyntaxError, ZeroDivisionError):
        return None
    return steps, values, inputs, literals


def compile_program(row):
    """Infer simple data flow by unambiguous numeric references, anchored in question digits."""
    trace = _verified_trace(row)
    if trace is None:
        return None
    steps, values, inputs, literals = trace
    gold = values[-1]

    last = len(steps) - 1
    for literal, count in inputs.items():
        if count != 1 or literals[literal] != 1:
            continue
        try:
            a, b, signature = _affine(steps, last, literal=literal)
        except ValueError:
            continue
        if not a or a * literal + b != gold:
            continue
        for pivot in sorted(range(last), key=lambda i: abs(i - (last - 1) / 2)):
            try:
                c, d, _ = _affine(steps, pivot, literal=literal)
                e, f, _ = _affine(steps, last, pivot=pivot)
            except ValueError:
                continue
            if not c or not e or (e * c, e * d + f) != (a, b):
                continue
            functions = ((c, d), (e, f), (a, b))
            if all(abs(v) <= MAX_COEFFICIENT for pair in functions for v in pair):
                return Program(functions, signature)
    return None


class ProgramReadout(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Sequential(nn.RMSNorm(hidden_size), nn.Linear(hidden_size, 2)) for _ in range(3)
        )

    def forward(self, states):
        return torch.stack([head(state.float()) for head, state in zip(self.heads, states, strict=True)], dim=1)


def program_loss(predictions, targets, valid):
    """Fit f, g and h, and require predictions to satisfy h(x) = g(f(x))."""
    predictions = predictions.float()
    coefficient_loss = F.smooth_l1_loss(predictions, targets, reduction="none").mean(dim=(1, 2))
    raw = predictions.clamp(-math.asinh(MAX_COEFFICIENT), math.asinh(MAX_COEFFICIENT)).sinh()
    a, b = (component.unsqueeze(-1) for component in raw.unbind(-1))
    points = raw.new_tensor([-1., 2.])
    first = a[:, 0] * points + b[:, 0]
    composed = a[:, 1] * first + b[:, 1]
    whole = a[:, 2] * points + b[:, 2]
    consistency_loss = F.smooth_l1_loss(composed.asinh(), whole.asinh(), reduction="none").mean(-1)
    per_example = coefficient_loss + 0.25 * consistency_loss
    stats = torch.stack(((coefficient_loss.detach() * valid).sum(),
                         (consistency_loss.detach() * valid).sum(), valid.sum()))
    return (per_example * valid).mean(), stats
