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

    def _is_valid(self, row, col, num):
        """Return True if placing num at (row, col) breaks no Sudoku rule."""
        # Row check
        if num in self.board[row]:
            return False

        # Column check
        if num in (self.board[r][col] for r in range(9)):
            return False

        # 3x3 box check
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if self.board[r][c] == num:
                    return False

        return True

    def _find_empty(self):
        """Return (row, col) of the next empty cell, or None if the board is full."""
        for r in range(9):
            for c in range(9):
                if self.board[r][c] == 0:
                    return r, c
        return None

    # ------------------------------------------------------------------
    # Backtracking solver
    # ------------------------------------------------------------------

    def solve(self):
        """Solve the board in-place using recursive backtracking.

        Returns True if a solution was found, False if the puzzle is unsolvable.
        """
        empty = self._find_empty()
        if empty is None:
            return True  # No empty cells — board is complete

        row, col = empty

        for num in range(1, 10):
            if self._is_valid(row, col, num):
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
