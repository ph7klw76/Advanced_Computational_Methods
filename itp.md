tested itp generated from https://atb.uq.edu.au/

```python

#!/usr/bin/env python3
"""
Update a GROMACS .itp file charges using RESP charges extracted from a text output
(e.g., photocatalysis.out). Keeps the decimal places of the .itp charge column and
enforces overall charge neutrality (to the nearest integer net charge inferred from
the original .itp).

Usage:
  python update_itp_resp.py -i 3.itp -o photocatalysis.out -out 3_resp_updated.itp

Notes:
- Only updates the [ atoms ] section.
- Preserves inline comments after ';' on each line.
- Tries to auto-detect a RESP charge block in the output and picks a block whose
  number of charges matches the number of atoms in the .itp.
"""

from __future__ import annotations
import argparse
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class AtomLine:
    raw: str                   # full original line including newline
    is_atom: bool              # whether this is an atom record line in [ atoms ]
    fields: Optional[List[str]]  # split fields for atom line (no comment)
    comment: str               # comment including leading ';' if present
    charge_str: Optional[str]  # original charge token (string)
    charge_decimals: Optional[int]  # decimals in original charge token


def split_comment(line: str) -> Tuple[str, str]:
    """Split a line into (main_part, comment_part). Comment includes leading ';'."""
    if ";" in line:
        main, comment = line.split(";", 1)
        return main.rstrip("\n"), ";" + comment.rstrip("\n")
    return line.rstrip("\n"), ""


def count_decimals(num_token: str) -> int:
    """Count decimal places in a numeric token like -0.123."""
    if "." in num_token:
        return len(num_token.split(".", 1)[1])
    return 0


def parse_itp_atoms(itp_path: str) -> Tuple[List[str], List[AtomLine], int]:
    """
    Parse .itp file, identify [ atoms ] section lines and atom records.
    Returns:
      - all lines of file
      - parsed AtomLine list for lines within [ atoms ] section (others are is_atom=False)
      - number of atom records (N)
    """
    with open(itp_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    in_atoms = False
    atom_lines: List[AtomLine] = []
    n_atoms = 0

    section_header_re = re.compile(r"^\s*\[\s*([A-Za-z0-9_]+)\s*\]\s*$")

    for line in lines:
        main, comment = split_comment(line)

        m = section_header_re.match(main.strip())
        if m:
            in_atoms = (m.group(1).lower() == "atoms")
            atom_lines.append(AtomLine(raw=line, is_atom=False, fields=None,
                                       comment=comment, charge_str=None, charge_decimals=None))
            continue

        if not in_atoms:
            atom_lines.append(AtomLine(raw=line, is_atom=False, fields=None,
                                       comment=comment, charge_str=None, charge_decimals=None))
            continue

        # Inside [ atoms ] section
        stripped = main.strip()
        if stripped == "" or stripped.startswith(("#", ";")):
            atom_lines.append(AtomLine(raw=line, is_atom=False, fields=None,
                                       comment=comment, charge_str=None, charge_decimals=None))
            continue

        # Attempt to parse an atom record line:
        # GROMACS typical fields: nr type resnr residue atom cgnr charge mass
        fields = stripped.split()
        if len(fields) < 7:
            # Not a standard atom line; keep unchanged
            atom_lines.append(AtomLine(raw=line, is_atom=False, fields=None,
                                       comment=comment, charge_str=None, charge_decimals=None))
            continue

        charge_token = fields[6]  # 7th field is charge
        # Validate charge token looks numeric
        try:
            float(charge_token)
        except ValueError:
            atom_lines.append(AtomLine(raw=line, is_atom=False, fields=None,
                                       comment=comment, charge_str=None, charge_decimals=None))
            continue

        n_atoms += 1
        atom_lines.append(AtomLine(raw=line, is_atom=True, fields=fields,
                                   comment=comment, charge_str=charge_token,
                                   charge_decimals=count_decimals(charge_token)))

    return lines, atom_lines, n_atoms


def infer_target_net_charge_from_itp(atom_lines: List[AtomLine]) -> int:
    """Infer the intended net charge as nearest integer from original .itp."""
    total = 0.0
    for al in atom_lines:
        if al.is_atom and al.fields is not None:
            total += float(al.fields[6])
    return int(round(total))


def most_common_decimal_places(atom_lines: List[AtomLine]) -> int:
    """Pick the most common decimal places used in the .itp charge column."""
    counts = {}
    for al in atom_lines:
        if al.is_atom and al.charge_decimals is not None:
            counts[al.charge_decimals] = counts.get(al.charge_decimals, 0) + 1
    if not counts:
        return 3
    return max(counts.items(), key=lambda kv: kv[1])[0]


def extract_candidate_charge_blocks(text: str) -> List[List[float]]:
    """
    Extract multiple candidate blocks of charges from a text output.

    Strategy:
    - Find regions around keywords like 'RESP' and 'charge'
    - Collect numeric charge-like tokens in those regions
    - Also scan the whole file for tables of (index, charge) patterns

    Returns a list of candidate charge lists.
    """
    candidates: List[List[float]] = []

    lines = text.splitlines()

    # Helper: pull floats from lines with patterns: "idx  charge" or "idx  atom  charge"
    idx_charge_re = re.compile(
        r"^\s*(\d+)\s+(?:[A-Za-z0-9_.-]+\s+)?([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$"
    )
    idx_atom_charge_re = re.compile(
        r"^\s*(\d+)\s+([A-Za-z]{1,3}\d*|[A-Za-z]{1,3})\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$"
    )

    # Scan for contiguous blocks of index/charge-ish lines
    current: List[float] = []
    last_idx = None
    for ln in lines:
        m2 = idx_atom_charge_re.match(ln)
        m1 = idx_charge_re.match(ln) if not m2 else None

        if m2:
            idx = int(m2.group(1))
            q = float(m2.group(3))
        elif m1:
            idx = int(m1.group(1))
            q = float(m1.group(2))
        else:
            # break block
            if len(current) >= 5:  # small threshold to ignore tiny fragments
                candidates.append(current)
            current = []
            last_idx = None
            continue

        # if indices are mostly increasing, treat as same block; otherwise start new
        if last_idx is not None and idx <= last_idx and len(current) >= 5:
            candidates.append(current)
            current = []

        current.append(q)
        last_idx = idx

    if len(current) >= 5:
        candidates.append(current)

    # Keyword-window approach: around "RESP" & "charge"
    key_re = re.compile(r"resp", re.IGNORECASE)
    charge_re = re.compile(r"charge", re.IGNORECASE)
    key_lines = [i for i, ln in enumerate(lines) if key_re.search(ln) and charge_re.search(ln)]
    for i in key_lines:
        # Take a window after the keyword line
        window = lines[i:i + 300]
        block: List[float] = []
        for ln in window:
            m2 = idx_atom_charge_re.match(ln)
            m1 = idx_charge_re.match(ln) if not m2 else None
            if m2:
                block.append(float(m2.group(3)))
            elif m1:
                block.append(float(m1.group(2)))
        if len(block) >= 5:
            candidates.append(block)

    return candidates


def choose_charge_block(candidates: List[List[float]], n_atoms: int) -> List[float]:
    """Choose the best candidate block matching n_atoms; prefer the last matching one."""
    matches = [c for c in candidates if len(c) == n_atoms]
    if matches:
        return matches[-1]

    # If no exact match, try the closest length (warn-like behavior via exception message)
    if not candidates:
        raise ValueError("No charge-like blocks found in the output file.")

    closest = min(candidates, key=lambda c: abs(len(c) - n_atoms))
    raise ValueError(
        f"No charge block length matched n_atoms={n_atoms}. Closest block length={len(closest)}."
    )


def round_and_neutralize(unrounded: List[float], target_net_charge_int: int, decimals: int) -> List[float]:
    """
    Round charges to 'decimals' and adjust by minimal-cost steps of 10^-decimals
    to hit target integer net charge.
    """
    unit = 10 ** (-decimals)
    rounded = [round(q, decimals) for q in unrounded]

    current_sum = sum(rounded)
    diff = target_net_charge_int - current_sum

    # Convert diff into discrete steps (integer number of +/- unit moves)
    steps = int(round(diff / unit))
    if steps == 0:
        return rounded

    # Greedy minimal-cost adjustment: each step picks the atom where applying +/-unit
    # changes the rounded value closest to the unrounded original.
    for _ in range(abs(steps)):
        direction = 1 if steps > 0 else -1  # +unit or -unit

        best_i = None
        best_cost = None

        for i, (q0, qr) in enumerate(zip(unrounded, rounded)):
            q_new = qr + direction * unit
            # cost: deviation from unrounded original
            cost = abs(q_new - q0)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_i = i

        if best_i is None:
            break
        rounded[best_i] = round(rounded[best_i] + direction * unit, decimals)

    # Final sanity: ensure exact match (within float wiggle)
    final_sum = sum(rounded)
    if round(final_sum - target_net_charge_int, decimals) != 0:
        # Last resort: force-correct the single best atom by remaining diff
        remaining = target_net_charge_int - final_sum
        force_steps = int(round(remaining / unit))
        if force_steps != 0:
            # Apply all remaining to the atom with smallest cost
            direction = 1 if force_steps > 0 else -1
            best_i, best_cost = None, None
            for i, (q0, qr) in enumerate(zip(unrounded, rounded)):
                q_new = qr + force_steps * unit
                cost = abs(q_new - q0)
                if best_cost is None or cost < best_cost:
                    best_cost, best_i = cost, i
            if best_i is not None:
                rounded[best_i] = round(rounded[best_i] + force_steps * unit, decimals)

    return rounded


def format_charge(q: float, decimals: int) -> str:
    """Format with fixed decimals (including trailing zeros)."""
    return f"{q:.{decimals}f}"


def update_itp_charges(
    itp_path: str,
    out_path: str,
    resp_out_path: str,
) -> None:
    # Parse itp
    _, atom_lines, n_atoms = parse_itp_atoms(itp_path)
    if n_atoms == 0:
        raise ValueError("No atom records found in [ atoms ] section of the .itp file.")

    target_net = infer_target_net_charge_from_itp(atom_lines)
    decimals = most_common_decimal_places(atom_lines)

    # Read output & extract charges
    with open(resp_out_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    candidates = extract_candidate_charge_blocks(text)
    resp_charges = choose_charge_block(candidates, n_atoms)

    # Round and enforce neutrality
    new_charges = round_and_neutralize(resp_charges, target_net, decimals)

    # Write updated itp
    new_lines: List[str] = []
    charge_idx = 0

    for al in atom_lines:
        if not al.is_atom or al.fields is None:
            new_lines.append(al.raw)
            continue

        fields = list(al.fields)
        old_charge = float(fields[6])
        fields[6] = format_charge(new_charges[charge_idx], decimals)
        charge_idx += 1

        # Preserve whether mass exists, and keep the comment as-is
        # Reconstruct with a clean, aligned-ish spacing
        # (If you need exact original spacing, that becomes a more complex "surgical replace".)
        main_part = "  ".join(fields)
        if al.comment:
            main_part = f"{main_part}  {al.comment}"
        new_lines.append(main_part + "\n")

    if charge_idx != n_atoms:
        raise RuntimeError("Internal error: not all atom charges were updated.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    # Print a tiny report
    new_sum = sum(new_charges)
    print("Updated .itp written to:", out_path)
    print("Atoms updated:", n_atoms)
    print("Decimal places enforced:", decimals)
    print("Target net charge (int):", target_net)
    print("Final net charge (rounded):", f"{new_sum:.{decimals}f}")


def main():
    ap = argparse.ArgumentParser(description="Update .itp charges using RESP charges from an output file.")
    ap.add_argument("-i", "--itp", required=True, help="Input .itp file (e.g., 3.itp)")
    ap.add_argument("-o", "--out", required=True, help="RESP output file (e.g., photocatalysis.out)")
    ap.add_argument("-out", "--itp_out", required=True, help="Output updated .itp filename")
    args = ap.parse_args()

    update_itp_charges(args.itp, args.itp_out, args.out)

if __name__ == "__main__":
    main()
```
