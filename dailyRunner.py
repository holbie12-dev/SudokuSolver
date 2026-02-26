"""
dailyRunner.py — end-to-end daily Sudoku pipeline

Steps
-----
1. Scrape  : download today's puzzle from sudoku.com (skipped if already saved)
2. Extract : use the CNN model to read the 9×9 board from the image
3. Solve   : backtracking solver with live visualisation
"""

import os
import argparse
import datetime
import pytz

from webScraper      import SudokuScraper
from sudokuExtractor import SudokuExtractor
from sudokuVisualiser import SudokuVisualiser, SPEEDS


PUZZLE_FOLDER = 'DailySudokuChallenges'
TIMEZONE      = 'Australia/Sydney'


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _banner(step, text):
    width = 60
    print(f'\n{"-" * width}')
    print(f'  Step {step}: {text}')
    print(f'{"-" * width}')


def _today_image_path():
    """Return the expected path for today's puzzle PNG (Sydney time)."""
    tz   = pytz.timezone(TIMEZONE)
    date = datetime.datetime.now(tz).strftime('%Y-%m-%d')
    return os.path.join(PUZZLE_FOLDER, f'sudoku_{date}.png')


# ----------------------------------------------------------------------
# Pipeline steps
# ----------------------------------------------------------------------

def step_scrape(skip=False):
    """Download today's puzzle. Returns the saved image path."""
    _banner(1, 'Scraping today\'s puzzle')

    image_path = _today_image_path()

    if os.path.exists(image_path):
        print(f'  Today\'s image already exists - skipping scrape.\n  -> {image_path}')
        return image_path

    if skip:
        # --skip-scrape passed but no image for today — fall back to most recent
        import glob
        existing = sorted(glob.glob(os.path.join(PUZZLE_FOLDER, '*.png')))
        if existing:
            print(f'  --skip-scrape: no image for today, using most recent.\n  -> {existing[-1]}')
            return existing[-1]
        raise FileNotFoundError(
            f'No puzzle images found in {PUZZLE_FOLDER}/ and --skip-scrape was set.'
        )

    os.makedirs(PUZZLE_FOLDER, exist_ok=True)
    scraper    = SudokuScraper()
    image_path = scraper.get_canvas_image()
    return image_path


def step_extract(image_path, model_path):
    """Read the board from the puzzle image. Returns a 9×9 list."""
    _banner(2, 'Extracting board from image')

    extractor = SudokuExtractor(model_path=model_path)
    board     = extractor.extract_board(image_path=image_path)

    print('\n  Extracted board:')
    # Indent the print_board output
    import io, sys
    buf = io.StringIO()
    old_stdout, sys.stdout = sys.stdout, buf
    SudokuExtractor.print_board(board)
    sys.stdout = old_stdout
    for line in buf.getvalue().splitlines():
        print(f'  {line}')

    return board


def step_solve(board, speed):
    """Launch the visualised backtracking solver."""
    _banner(3, 'Solving & visualising')
    print(f'  Speed: {speed}  (change with --speed slow|medium|fast|instant)\n')

    vis = SudokuVisualiser(board, delay=SPEEDS[speed])
    vis.run()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Daily Sudoku pipeline: scrape → extract → solve & visualise.'
    )
    ap.add_argument(
        '--speed', default='fast', choices=SPEEDS.keys(),
        help='Visualisation speed (default: fast)'
    )
    ap.add_argument(
        '--model', default='models/model.keras', type=str,
        help='Path to the trained digit classifier model.'
    )
    ap.add_argument(
        '--skip-scrape', action='store_true',
        help='Skip the web scrape and use the most recent saved image.'
    )
    args = ap.parse_args()

    print('\n' + '=' * 60)
    print('         Daily Sudoku Solver Pipeline')
    print('=' * 60)

    try:
        image_path = step_scrape(skip=args.skip_scrape)
        board      = step_extract(image_path, args.model)
        step_solve(board, args.speed)
    except Exception as e:
        print(f'\n  ERROR: {e}')
        raise
