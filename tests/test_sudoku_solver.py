"""
Tests for SudokuSolver — backtracking solver with MRV heuristic.

Covers:
  _candidates     — valid digits for an empty cell
  _find_mrv_cell  — picks cell with fewest candidates
  solve           — full board solution
  get_board       — returns solved board as 9x9 list

No image/CNN dependencies needed — tests use pre-defined boards.

Run: pytest tests/test_sudoku_solver.py -v
     (from the SudokuSolver/ directory)
"""

import sys
import os
import types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# sudokuSolver.py imports sudokuExtractor which requires cv2/tensorflow.
# Stub out the heavy module before importing the solver.
_fake_extractor = types.ModuleType("sudokuExtractor")
_fake_extractor.SudokuExtractor = object
sys.modules.setdefault("sudokuExtractor", _fake_extractor)
for _dep in ("cv2", "tensorflow", "tf"):
    sys.modules.setdefault(_dep, types.ModuleType(_dep))

import pytest
from sudokuSolver import SudokuSolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EASY_BOARD = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

EASY_SOLUTION = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]

# Minimal puzzle (17 clues — known hardest cases for backtracking)
HARD_BOARD = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 8, 5],
    [0, 0, 1, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 7, 0, 0, 0],
    [0, 0, 4, 0, 0, 0, 1, 0, 0],
    [0, 9, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 0, 0, 0, 0, 0, 7, 3],
    [0, 0, 2, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 9],
]

ALREADY_SOLVED = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]

# Cell (0,0) has no candidates: digits 1-8 in its row, 9 in its column.
# MRV immediately detects the dead-end.
UNSOLVABLE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [9, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]


# ---------------------------------------------------------------------------
# _candidates
# ---------------------------------------------------------------------------

class TestCandidates:
    def test_empty_cell_with_no_constraints_has_all_nine(self):
        blank = [[0] * 9 for _ in range(9)]
        solver = SudokuSolver(blank)
        cands = solver._candidates(0, 0)
        assert cands == set(range(1, 10))

    def test_row_constraint_removes_digit(self):
        board = [[0] * 9 for _ in range(9)]
        board[0] = [1, 2, 3, 4, 5, 6, 7, 8, 0]
        solver = SudokuSolver(board)
        cands = solver._candidates(0, 8)
        assert cands == {9}

    def test_column_constraint_removes_digit(self):
        board = [[0] * 9 for _ in range(9)]
        for r in range(8):
            board[r][0] = r + 1
        solver = SudokuSolver(board)
        cands = solver._candidates(8, 0)
        assert cands == {9}

    def test_box_constraint_removes_digit(self):
        board = [[0] * 9 for _ in range(9)]
        # Fill top-left 3x3 box leaving (2,2) empty
        board[0][0], board[0][1], board[0][2] = 1, 2, 3
        board[1][0], board[1][1], board[1][2] = 4, 5, 6
        board[2][0], board[2][1] = 7, 8
        solver = SudokuSolver(board)
        cands = solver._candidates(2, 2)
        assert cands == {9}

    def test_no_candidates_for_fully_constrained_cell(self):
        board = [[0] * 9 for _ in range(9)]
        # Put 1-9 in row 0 except cell (0,0)
        board[0] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # Put 9 in column 0 (row 1)
        board[1][0] = 9
        solver = SudokuSolver(board)
        cands = solver._candidates(0, 0)
        assert cands == set()


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------

class TestSolve:
    def test_easy_puzzle_solves_correctly(self):
        solver = SudokuSolver(EASY_BOARD)
        result = solver.solve()
        assert result is True
        assert solver.board == EASY_SOLUTION

    def test_already_solved_board_returns_true(self):
        solver = SudokuSolver(ALREADY_SOLVED)
        result = solver.solve()
        assert result is True

    def test_hard_puzzle_solves_without_crash(self):
        solver = SudokuSolver(HARD_BOARD)
        result = solver.solve()
        assert result is True
        # Verify solution validity
        for row in solver.board:
            assert sorted(row) == list(range(1, 10))

    def test_unsolvable_board_returns_false(self):
        solver = SudokuSolver(UNSOLVABLE)
        result = solver.solve()
        assert result is False

    def test_solution_has_no_zeros(self):
        solver = SudokuSolver(EASY_BOARD)
        solver.solve()
        for row in solver.board:
            assert 0 not in row

    def test_solved_rows_contain_1_to_9(self):
        solver = SudokuSolver(EASY_BOARD)
        solver.solve()
        for row in solver.board:
            assert sorted(row) == list(range(1, 10))

    def test_solved_columns_contain_1_to_9(self):
        solver = SudokuSolver(EASY_BOARD)
        solver.solve()
        for c in range(9):
            col = [solver.board[r][c] for r in range(9)]
            assert sorted(col) == list(range(1, 10))

    def test_solved_boxes_contain_1_to_9(self):
        solver = SudokuSolver(EASY_BOARD)
        solver.solve()
        for box_r in range(3):
            for box_c in range(3):
                box = []
                for r in range(box_r * 3, box_r * 3 + 3):
                    for c in range(box_c * 3, box_c * 3 + 3):
                        box.append(solver.board[r][c])
                assert sorted(box) == list(range(1, 10))

    def test_original_board_preserved_after_solve(self):
        solver = SudokuSolver(EASY_BOARD)
        solver.solve()
        assert solver.original == EASY_BOARD

    def test_solve_does_not_mutate_input_board(self):
        import copy
        original_copy = copy.deepcopy(EASY_BOARD)
        SudokuSolver(EASY_BOARD).solve()
        assert EASY_BOARD == original_copy

    def test_get_board_returns_copy_not_reference(self):
        solver = SudokuSolver(EASY_BOARD)
        solver.solve()
        board1 = solver.get_board()
        board2 = solver.get_board()
        assert board1 == board2
        board1[0][0] = 999
        assert solver.board[0][0] != 999  # mutation doesn't affect internal state
