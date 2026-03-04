#!/usr/bin/env python3
"""
Parse an SSA MLIR file and return a DAG representation of the data-flow graph.

Node: an SSA value (%name) or a block argument.
Edge: (producer, consumer) meaning the producer's value is used as an operand
      by the operation that defines the consumer.

Ops with no result (e.g. func.return) are represented as a synthetic node
named "__return__" (or "__return_N__" if there are multiple).
"""

import re
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Node:
    name: str                        # SSA value name, e.g. "%lhs" or "__return_0__"
    op: str                          # operation name, e.g. "transfer.get"
    operands: list[str]              # list of SSA value names used as inputs
    result_type: Optional[str]       # result type (None for void ops / block args)
    attrs: dict                      # parsed attribute dict (best-effort)
    is_block_arg: bool = False

    def to_dict(self) -> dict:
        d = {
            "op": self.op,
            "operands": self.operands,
            "result_type": self.result_type,
            "attrs": self.attrs,
        }
        if self.is_block_arg:
            d["is_block_arg"] = True
        return d


@dataclass
class DAG:
    nodes: dict[str, Node] = field(default_factory=dict)   # name -> Node
    edges: list[tuple[str, str]] = field(default_factory=list)  # (producer, consumer)

    def to_dict(self) -> dict:
        return {
            "nodes": {name: node.to_dict() for name, node in self.nodes.items()},
            "edges": [{"from": src, "to": dst} for src, dst in self.edges],
        }


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches:  %result = "op.name"(%a, %b) {attrs} : (types) -> result_type
OP_WITH_RESULT = re.compile(
    r'^\s*(%\S+?)\s*=\s*"([^"]+)"\s*\(([^)]*)\)\s*((?:\{[^}]*\})?)\s*:\s*([^>]+?->.*)'
)

# Matches:  "op.name"(%a, %b) {attrs} : (types) -> ()
OP_NO_RESULT = re.compile(
    r'^\s*"([^"]+)"\s*\(([^)]*)\)\s*((?:\{[^}]*\})?)\s*:\s*([^>]+?->.*)'
)

# Matches block argument line:  ^label(%arg1 : type1, %arg2 : type2):
BLOCK_ARGS = re.compile(r'^\s*\^[^(]*\(([^)]+)\)\s*:')

# Extracts %names from an operand list string
OPERAND_NAMES = re.compile(r'%\S+?(?=\s*[,)]|\s*$)')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_operands(operand_str: str) -> list[str]:
    """Return list of %name tokens from a comma-separated operand string."""
    return re.findall(r'%[\w.]+', operand_str)


def parse_attrs(attr_str: str) -> dict:
    """Best-effort parse of {key = value, ...} attribute block."""
    attrs: dict = {}
    attr_str = attr_str.strip().lstrip('{').rstrip('}').strip()
    if not attr_str:
        return attrs
    # Split on commas that are NOT inside angle brackets or nested braces
    depth = 0
    current = []
    for ch in attr_str:
        if ch in '<({':
            depth += 1
        elif ch in '>)}':
            depth -= 1
        if ch == ',' and depth == 0:
            part = ''.join(current).strip()
            if '=' in part:
                k, _, v = part.partition('=')
                attrs[k.strip()] = v.strip()
            current = []
        else:
            current.append(ch)
    part = ''.join(current).strip()
    if '=' in part:
        k, _, v = part.partition('=')
        attrs[k.strip()] = v.strip()
    return attrs


def extract_result_type(type_sig: str) -> Optional[str]:
    """
    From a type signature like '(!transfer.integer, i1) -> !transfer.integer'
    return the result type string, or None if result is '()'.
    """
    arrow_idx = type_sig.rfind('->')
    if arrow_idx == -1:
        return None
    result = type_sig[arrow_idx + 2:].strip()
    if result == '()':
        return None
    return result


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_mlir(text: str) -> DAG:
    dag = DAG()
    return_counter = 0

    for line in text.splitlines():
        # Skip empty lines and pure comments
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            continue

        # ---- Block arguments (%lhs : type, %rhs : type) ----
        m = BLOCK_ARGS.match(line)
        if m:
            args_str = m.group(1)
            # Each argument is "  %name : type"
            for arg_part in args_str.split(','):
                arg_part = arg_part.strip()
                am = re.match(r'(%[\w.]+)\s*:\s*(.+)', arg_part)
                if am:
                    name, typ = am.group(1), am.group(2).strip()
                    node = Node(
                        name=name,
                        op="block_arg",
                        operands=[],
                        result_type=typ,
                        attrs={},
                        is_block_arg=True,
                    )
                    dag.nodes[name] = node
            continue

        # ---- Op with result:  %x = "op"(...) ----
        m = OP_WITH_RESULT.match(line)
        if m:
            result_name = m.group(1)
            op_name = m.group(2)
            operand_str = m.group(3)
            attr_str = m.group(4)
            type_sig = m.group(5)

            operands = extract_operands(operand_str)
            attrs = parse_attrs(attr_str)
            result_type = extract_result_type(type_sig)

            node = Node(
                name=result_name,
                op=op_name,
                operands=operands,
                result_type=result_type,
                attrs=attrs,
            )
            dag.nodes[result_name] = node
            for src in operands:
                if src in dag.nodes or src not in dag.nodes:
                    # Always add the edge; dangling refs may be block args not yet seen
                    dag.edges.append((src, result_name))
            continue

        # ---- Op with no result:  "op"(...) ----
        m = OP_NO_RESULT.match(line)
        if m:
            op_name = m.group(1)
            operand_str = m.group(2)
            attr_str = m.group(3)
            type_sig = m.group(4)

            # Skip the outer func.func wrapper (it has a block body, not simple operands)
            if op_name == "func.func":
                continue

            operands = extract_operands(operand_str)
            attrs = parse_attrs(attr_str)

            synthetic_name = f"__return_{return_counter}__"
            return_counter += 1

            node = Node(
                name=synthetic_name,
                op=op_name,
                operands=operands,
                result_type=None,
                attrs=attrs,
            )
            dag.nodes[synthetic_name] = node
            for src in operands:
                dag.edges.append((src, synthetic_name))
            continue

    return dag


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert MLIR file to DAG")
    parser.add_argument(
        "input",
        type=Path,
        help="MLIR file to convert to DAG"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="format output as pretty JSON"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output file path"
    )
    args = parser.parse_args()

    text = args.input.read_text()
    dag = parse_mlir(text)

    indent = 2 if args.pretty else None
    out = json.dumps(dag.to_dict(), indent=indent)

    if args.output:
        args.output.write_text(out)
    else:
        print(out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
