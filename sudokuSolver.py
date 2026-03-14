import argparse
from sudokuExtractor import SudokuExtractor


class SudokuSolver:
    def __init__(self, board):
        # Work on a deep copy so the original is preserved
        self.board = [row[:] for row in board]
        self.original = [row[:] for row in board]

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _candidates(self, row, col):
        """Return the set of valid digits for the empty cell at (row, col)."""
        used = set(self.board[row])
        used.update(self.board[r][col] for r in range(9))
        box_row, box_col = (row // 3) * 3, (col // 3) * 3
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                used.add(self.board[r][c])
        return set(range(1, 10)) - used

    def _find_mrv_cell(self):
        """Return (row, col, candidates) for the empty cell with the fewest
        valid candidates (Minimum Remaining Values heuristic), or None if the
        board is full.  Returns immediately if a cell with only one candidate
        is found, or signals a dead-end if a cell has zero candidates.
        """
        best_pos = None
        best_candidates = None
        best_count = 10  # More than the maximum possible (9)

        for r in range(9):
            for c in range(9):
                if self.board[r][c] != 0:
                    continue
                cands = self._candidates(r, c)
                count = len(cands)
                if count == 0:
                    return r, c, cands  # Dead-end: no valid digit exists
                if count < best_count:
                    best_count = count
                    best_pos = (r, c)
                    best_candidates = cands
                    if count == 1:
                        break  # Cannot do better — skip the rest of the search
            if best_count == 1:
                break

        if best_pos is None:
            return None  # Board is complete
        r, c = best_pos
        return r, c, best_candidates

    # ------------------------------------------------------------------
    # Backtracking solver
    # ------------------------------------------------------------------

    def solve(self):
        """Solve the board in-place using backtracking with the MRV heuristic.

        Picks the empty cell with fewest valid candidates at each step, which
        minimises branching and reduces total backtracks.  Returns True if a
        solution was found, False if the puzzle is unsolvable.
        """
        result = self._find_mrv_cell()
        if result is None:
            return True  # No empty cells — board is complete

        row, col, candidates = result
        if not candidates:
            return False  # Dead-end: no valid digit for this cell

        for num in candidates:
            self.board[row][col] = num

            if self.solve():
                return True

            # Backtrack
            self.board[row][col] = 0

        return False  # Trigger backtrack in caller

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    @staticmethod
    def _format_board(board, original=None):
        """Return the board as a formatted string.

        If original is supplied, solved cells are marked with square brackets.
        """
        divider = "+-------+-------+-------+"
        lines = ["\n", divider]
        for i, row in enumerate(board):
            line = "| "
            for j, val in enumerate(row):
                if val == 0:
                    cell = "."
                elif original and original[i][j] == 0:
                    cell = f"[{val}]"   # Solved cell
                else:
                    cell = f" {val} "   # Given clue
                line += cell + " "
                if (j + 1) % 3 == 0 and j < 8:
                    line += "| "
            line += "|"
            lines.append(line)
            if (i + 1) % 3 == 0:
                lines.append(divider)
        return "\n".join(lines)

    def print_solution(self):
        """Print the original board and the solved board side by side."""
        print("\nOriginal puzzle:")
        print(self._format_board(self.original))
        print("\nSolved puzzle:  ([x] = filled by solver)")
        print(self._format_board(self.board, original=self.original))

    def get_board(self):
        """Return the current (solved) board as a 9x9 list."""
        return [row[:] for row in self.board]


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Solve a Sudoku puzzle extracted from an image.")
    ap.add_argument(
        "--image", default=None, type=str,
        help="Path to a puzzle PNG. Defaults to the most recent in DailySudokuChallenges/."
    )
    ap.add_argument(
        "--model", default="models/model.keras", type=str,
        help="Path to the trained digit classifier model."
    )
    args = ap.parse_args()

    # Step 1: Extract the board from the image
    extractor = SudokuExtractor(model_path=args.model)
    board = extractor.extract_board(image_path=args.image)

    print("\nExtracted board:")
    SudokuExtractor.print_board(board)

    # Step 2: Solve
    solver = SudokuSolver(board)
    if solver.solve():
        solver.print_solution()
    else:
        print("\nNo solution exists for this puzzle.")
        print("The extracted board may contain recognition errors.")
