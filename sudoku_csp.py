"""
CSP-Based Sudoku Solver
========================
Techniques used:
  - AC-3 (Arc Consistency Algorithm 3)  — constraint propagation
  - Forward Checking                     — prune domains after each assignment
  - Backtracking Search                  — systematic search with MRV heuristic

Input  : .txt file with 9 lines of 9 digits (0 = empty cell)
Output : Solved board + statistics (calls, failures)

Author : CSP Assignment — Spring 2026
"""

import copy
import time


# ---------------------------------------------------------------------------
# Board loading and display
# ---------------------------------------------------------------------------

def load_board(filename):
    """Read a sudoku board from a text file. Returns a 9x9 list of ints."""
    board = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                row = [int(ch) for ch in line]
                board.append(row)
    if len(board) != 9 or any(len(r) != 9 for r in board):
        raise ValueError(f"Board in '{filename}' must be exactly 9x9.")
    return board


def print_board(board, title=""):
    """Pretty-print a 9x9 sudoku board with box separators."""
    if title:
        print(f"\n{'=' * 37}")
        print(f"  {title}")
        print(f"{'=' * 37}")
    print("+-------+-------+-------+")
    for r in range(9):
        if r in (3, 6):
            print("+-------+-------+-------+")
        row_str = "| "
        for c in range(9):
            val = board[r][c]
            row_str += (str(val) if val != 0 else ".") + " "
            if c in (2, 5):
                row_str += "| "
        print(row_str + "|")
    print("+-------+-------+-------+")


# ---------------------------------------------------------------------------
# CSP representation
# ---------------------------------------------------------------------------

def build_domains(board):
    """
    Build the initial domain for every cell.
    Filled cells have a single-element domain {value}.
    Empty cells start with {1..9}.
    """
    domains = {}
    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                domains[(r, c)] = {board[r][c]}
            else:
                domains[(r, c)] = set(range(1, 10))
    return domains


def get_peers(r, c):
    """
    Return all cells that share a row, column, or 3x3 box with (r, c).
    A cell is NOT its own peer.
    """
    peers = set()

    # same row
    for col in range(9):
        if col != c:
            peers.add((r, col))

    # same column
    for row in range(9):
        if row != r:
            peers.add((row, c))

    # same 3x3 box
    box_r, box_c = (r // 3) * 3, (c // 3) * 3
    for dr in range(3):
        for dc in range(3):
            nr, nc = box_r + dr, box_c + dc
            if (nr, nc) != (r, c):
                peers.add((nr, nc))

    return peers


# Precompute peers for every cell (avoids recomputing during search)
PEERS = {(r, c): get_peers(r, c) for r in range(9) for c in range(9)}


def get_arcs():
    """Return every directed arc (Xi, Xj) where Xi and Xj are peers."""
    arcs = []
    for cell in PEERS:
        for peer in PEERS[cell]:
            arcs.append((cell, peer))
    return arcs


# ---------------------------------------------------------------------------
# AC-3 — Arc Consistency
# ---------------------------------------------------------------------------

def ac3(domains):
    """
    AC-3 constraint propagation.
    Removes values from domains that cannot satisfy the ≠ constraint.
    Returns False if any domain becomes empty (unsolvable), True otherwise.
    Modifies domains IN PLACE.
    """
    queue = get_arcs()

    while queue:
        (xi, xj) = queue.pop(0)

        if revise(domains, xi, xj):
            if len(domains[xi]) == 0:
                return False          # domain wiped out — contradiction!
            # xi's domain shrank → re-check all arcs pointing to xi
            for xk in PEERS[xi]:
                if xk != xj:
                    queue.append((xk, xi))

    return True


def revise(domains, xi, xj):
    """
    Remove values from domain(xi) that have no valid match in domain(xj).
    For Sudoku the constraint is simply xi ≠ xj.
    Returns True if any value was removed.
    """
    revised = False
    for val in set(domains[xi]):       # iterate over a copy
        # val is only invalid if xj's entire domain equals {val}
        # meaning no other option exists for xj
        if domains[xj] == {val}:
            domains[xi].remove(val)
            revised = True
    return revised


# ---------------------------------------------------------------------------
# Forward Checking
# ---------------------------------------------------------------------------

def forward_check(domains, cell, value):
    """
    After assigning `value` to `cell`, remove `value` from all peer domains.
    Returns False if any peer domain becomes empty, True otherwise.
    Works on a COPY of domains (does not modify original).
    """
    new_domains = copy.deepcopy(domains)
    new_domains[cell] = {value}

    for peer in PEERS[cell]:
        new_domains[peer].discard(value)
        if len(new_domains[peer]) == 0:
            return None               # failure — domain wiped out

    return new_domains


# ---------------------------------------------------------------------------
# MRV Heuristic — variable selection
# ---------------------------------------------------------------------------

def select_unassigned_variable(domains, assignment):
    """
    MRV (Minimum Remaining Values):
    Pick the unassigned cell with the fewest legal values.
    Ties broken by cell position (top-left first).
    """
    unassigned = [cell for cell in domains if cell not in assignment]
    # sort by domain size (ascending), then by cell index for tie-breaking
    return min(unassigned, key=lambda cell: (len(domains[cell]), cell))


# ---------------------------------------------------------------------------
# Backtracking Search
# ---------------------------------------------------------------------------

def backtrack(assignment, domains, stats):
    """
    Recursive backtracking search.
    assignment : dict { (r,c): value } for already-assigned cells
    domains    : dict { (r,c): set of remaining values }
    stats      : dict tracking 'calls' and 'failures'
    Returns a complete assignment dict, or None on failure.
    """
    stats["calls"] += 1

    # Base case — all 81 cells assigned
    if len(assignment) == 81:
        return assignment

    # Choose the next variable using MRV
    cell = select_unassigned_variable(domains, assignment)

    # Try each value in the domain (sorted for determinism)
    for value in sorted(domains[cell]):

        # Forward checking — prune peers
        new_domains = forward_check(domains, cell, value)

        if new_domains is not None:
            # Assignment looks valid so far
            assignment[cell] = value

            # Apply AC-3 on top of forward checking for stronger pruning
            if ac3(new_domains):
                result = backtrack(assignment, new_domains, stats)
                if result is not None:
                    return result

            # This value didn't work — undo
            del assignment[cell]

    # No value worked — report failure and backtrack
    stats["failures"] += 1
    return None


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve(filename):
    """
    Full pipeline:
      1. Load board from file
      2. Build CSP domains
      3. Run AC-3 as preprocessing
      4. Run backtracking with forward checking + MRV
      5. Print results and stats
    """
    print(f"\n{'#' * 45}")
    print(f"  Solving: {filename}")
    print(f"{'#' * 45}")

    board = load_board(filename)
    print_board(board, title=f"Initial Board — {filename}")

    domains = build_domains(board)

    # --- Preprocessing with AC-3 ---
    print("\n[AC-3] Running initial constraint propagation...")
    if not ac3(domains):
        print("[AC-3] No solution exists after preprocessing.")
        return

    # Build the initial assignment from given clues
    assignment = {}
    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                assignment[(r, c)] = board[r][c]

    print(f"[AC-3] Done. Clues given: {len(assignment)}/81")

    # --- Backtracking Search ---
    stats = {"calls": 0, "failures": 0}

    print("[Search] Starting backtracking search...\n")
    start = time.time()
    result = backtrack(assignment, domains, stats)
    elapsed = time.time() - start

    # --- Results ---
    if result is None:
        print("No solution found.")
    else:
        # Convert assignment dict → 9x9 board for display
        solved_board = [[0] * 9 for _ in range(9)]
        for (r, c), val in result.items():
            solved_board[r][c] = val

        print_board(solved_board, title=f"Solved Board — {filename}")
        print(f"\n  Time taken      : {elapsed:.4f} seconds")
        print(f"  Backtrack calls : {stats['calls']}")
        print(f"  Backtrack fails : {stats['failures']}")

    return stats


# ---------------------------------------------------------------------------
# Entry point — solve all four boards
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    puzzles = ["easy.txt", "medium.txt", "hard.txt", "veryhard.txt"]

    all_stats = {}

    for puzzle in puzzles:
        try:
            s = solve(puzzle)
            all_stats[puzzle] = s
        except FileNotFoundError:
            print(f"\n[WARNING] File '{puzzle}' not found — skipping.")

    # --- Summary table ---
    print(f"\n\n{'=' * 55}")
    print(f"  SUMMARY — All Boards")
    print(f"{'=' * 55}")
    print(f"  {'Board':<15} {'Calls':>10} {'Failures':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10}")
    for puzzle, s in all_stats.items():
        if s:
            print(f"  {puzzle:<15} {s['calls']:>10} {s['failures']:>10}")
    print(f"{'=' * 55}")

    print("""
OBSERVATIONS:
  Easy      — Very few backtracks. AC-3 alone almost solves it.
  Medium    — Slightly more calls; forward checking handles most conflicts.
  Hard      — Noticeably more failures; MRV keeps search focused.
  Very Hard — Most calls/failures; demonstrates AC-3 + backtracking synergy.
""")
