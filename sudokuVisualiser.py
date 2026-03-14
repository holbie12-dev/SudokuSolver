import time
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sudokuExtractor import SudokuExtractor


# Colour palette
_BOX_BG   = ['#eaf3fb', '#d0e8f5']   # alternating 3x3 box backgrounds
_CLR_GIVEN     = '#1a2e4a'            # dark navy  — given clue text
_CLR_TRY       = '#1a6bcc'            # blue       — solver placing a digit
_CLR_BACKTRACK = '#cc2200'            # red        — digit about to be removed
_BG_TRY        = '#d4f5d4'            # light green — active cell (placing)
_BG_BACKTRACK  = '#fad4d4'            # light red  — active cell (backtracking)


SPEEDS = {
    'slow':    0.12,
    'medium':  0.03,
    'fast':    0.005,
    'instant': 0.0,
}


class SudokuVisualiser:
    def __init__(self, board, delay=0.03):
        self.board    = [row[:] for row in board]
        self.original = [row[:] for row in board]
        self.delay    = delay
        self.steps    = 0

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.patch.set_facecolor('#f0f4f8')
        self._build_canvas()
        plt.ion()
        plt.show()

    # ------------------------------------------------------------------
    # Canvas setup — create every patch and text object once
    # ------------------------------------------------------------------

    def _build_canvas(self):
        self.ax.set_xlim(0, 9)
        self.ax.set_ylim(0, 9)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # Cell background patches (one per cell, updated in-place during solve)
        self.patches = {}
        for r in range(9):
            for c in range(9):
                bg = _BOX_BG[((r // 3) * 3 + (c // 3)) % 2]
                rect = mpatches.Rectangle(
                    (c, 8 - r), 1, 1,
                    facecolor=bg, edgecolor='none', zorder=1
                )
                self.ax.add_patch(rect)
                self.patches[(r, c)] = rect

        # Grid lines — thin for cells, thick for 3x3 box borders
        for i in range(10):
            thick = (i % 3 == 0)
            lw    = 2.5 if thick else 0.5
            color = '#2c4a6e' if thick else '#9ab0c8'
            self.ax.plot([i, i], [0, 9], color=color, linewidth=lw, zorder=3)
            self.ax.plot([0, 9], [i, i], color=color, linewidth=lw, zorder=3)

        # Cell text objects (one per cell, updated in-place during solve)
        self.texts = {}
        for r in range(9):
            for c in range(9):
                val = self.board[r][c]
                txt = self.ax.text(
                    c + 0.5, (8 - r) + 0.5,
                    str(val) if val else '',
                    ha='center', va='center',
                    fontsize=20, color=_CLR_GIVEN, fontweight='bold',
                    zorder=4
                )
                self.texts[(r, c)] = txt

        self._set_title('Solving…')
        self.fig.canvas.draw()

    # ------------------------------------------------------------------
    # Cell update — change colour and text without rebuilding anything
    # ------------------------------------------------------------------

    def _update_cell(self, row, col, value, state='normal'):
        """Repaint a single cell and flush the canvas.

        state values:
          'normal'    — reset to default (given clue or empty)
          'try'       — solver is attempting this digit (green bg, blue text)
          'backtrack' — digit is wrong, about to be erased (red bg, red text)
        """
        box_bg = _BOX_BG[((row // 3) * 3 + (col // 3)) % 2]

        if state == 'try':
            bg, color, weight = _BG_TRY, _CLR_TRY, 'normal'
        elif state == 'backtrack':
            bg, color, weight = _BG_BACKTRACK, _CLR_BACKTRACK, 'normal'
        else:
            bg     = box_bg
            color  = _CLR_GIVEN if self.original[row][col] != 0 else _CLR_TRY
            weight = 'bold' if self.original[row][col] != 0 else 'normal'

        self.patches[(row, col)].set_facecolor(bg)
        t = self.texts[(row, col)]
        t.set_text(str(value) if value else '')
        t.set_color(color)
        t.set_fontweight(weight)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self.delay > 0:
            plt.pause(self.delay)

    def _set_title(self, msg, color='#1a2e4a'):
        self.fig.suptitle(msg, fontsize=13, color=color, fontweight='bold',
                          y=0.97)

    # ------------------------------------------------------------------
    # Backtracking solver (with visualisation hooks)
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
        valid candidates (MRV heuristic), or None if the board is full."""
        best_pos = None
        best_candidates = None
        best_count = 10

        for r in range(9):
            for c in range(9):
                if self.board[r][c] != 0:
                    continue
                cands = self._candidates(r, c)
                count = len(cands)
                if count == 0:
                    return r, c, cands  # Dead-end
                if count < best_count:
                    best_count = count
                    best_pos = (r, c)
                    best_candidates = cands
                    if count == 1:
                        break
            if best_count == 1:
                break

        if best_pos is None:
            return None  # Board complete
        r, c = best_pos
        return r, c, best_candidates

    def _solve(self):
        result = self._find_mrv_cell()
        if result is None:
            return True  # No empty cells — solved

        row, col, candidates = result
        if not candidates:
            return False  # Dead-end

        for num in candidates:
            self.board[row][col] = num
            self.steps += 1
            self._set_title(f'Steps: {self.steps:,}')
            self._update_cell(row, col, num, state='try')

            if self._solve():
                return True

            # Wrong — backtrack
            self._update_cell(row, col, num, state='backtrack')
            self.board[row][col] = 0
            self._update_cell(row, col, 0, state='normal')

        return False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self):
        print("Visualising backtracking solve — close the window when done.")
        start   = time.time()
        solved  = self._solve()
        elapsed = time.time() - start

        if solved:
            msg = f'Solved in {self.steps:,} steps ({elapsed:.2f}s)'
            print(f'\n{msg}')
            self._set_title(msg, color='#1a7a1a')
        else:
            print('\nNo solution found.')
            self._set_title('No solution found', color='#cc2200')

        self.fig.canvas.draw()
        plt.ioff()
        plt.show()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Visualise the backtracking Sudoku solver on a scraped puzzle image.'
    )
    ap.add_argument(
        '--image', default=None, type=str,
        help='Path to a puzzle PNG. Defaults to the most recent in DailySudokuChallenges/.'
    )
    ap.add_argument(
        '--model', default='models/model.keras', type=str,
        help='Path to the trained digit classifier model.'
    )
    ap.add_argument(
        '--speed', default='medium', choices=SPEEDS.keys(),
        help='Animation speed: slow | medium | fast | instant  (default: medium)'
    )
    args = ap.parse_args()

    # Step 1: extract the board from the image
    extractor = SudokuExtractor(model_path=args.model)
    board = extractor.extract_board(image_path=args.image)

    print('\nExtracted board:')
    SudokuExtractor.print_board(board)

    # Step 2: run the visualised solver
    vis = SudokuVisualiser(board, delay=SPEEDS[args.speed])
    vis.run()
