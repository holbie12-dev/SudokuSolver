# Sudoku Solver

An end-to-end pipeline that scrapes the daily Sudoku puzzle from [sudoku.com](https://sudoku.com/challenges/daily-sudoku), reads the board using a CNN digit classifier, and solves it with an animated backtracking algorithm.

---

## Pipeline Overview

```
webScraper.py          dataCleaning.py + modelTraining.py
(scrape puzzle PNG)    (train CNN digit classifier)
        |                          |
        |                   models/model.keras
        |                          |
        +---------> sudokuExtractor.py (read 9x9 board from image)
                            |
                    sudokuSolver.py / sudokuVisualiser.py
                    (solve + animate backtracking)
```

The `dailyRunner.py` script runs all steps automatically in sequence.

---

## Project Structure

```
SudokuSolver/
├── dailyRunner.py          # End-to-end pipeline entry point
├── webScraper.py           # Scrapes the daily puzzle as a PNG screenshot
├── sudokuExtractor.py      # CNN-based board reader (image → 9×9 grid)
├── sudokuSolver.py         # Backtracking solver (no visualisation)
├── sudokuVisualiser.py     # Backtracking solver with live animation
├── dataCleaning.py         # Preprocesses digit training images
├── modelTraining.py        # Builds, trains, and saves the CNN
├── DigitData/              # Training images organised by digit (Sample002–010)
├── DailySudokuChallenges/  # Scraped puzzle screenshots (auto-created)
└── models/                 # Saved trained models (auto-created)
```

---

## Quick Start

**1. Install dependencies**
```bash
pip install tensorflow opencv-python scikit-learn selenium webdriver-manager Pillow pytz numpy matplotlib
```

**2. Train the digit classifier** (one-time setup)
```bash
python modelTraining.py
```

**3. Run the daily pipeline**
```bash
python dailyRunner.py
```

This will scrape today's puzzle, extract the board, and open an animated solver window.

---

## Files

### `dailyRunner.py` — Main pipeline
Runs all three steps in sequence: scrape → extract → solve & visualise.

```bash
python dailyRunner.py [--speed slow|medium|fast|instant] [--model PATH] [--skip-scrape]
```

| Flag | Default | Description |
|---|---|---|
| `--speed` | `fast` | Visualisation speed |
| `--model` | `models/model.keras` | Path to trained model |
| `--skip-scrape` | off | Skip scraping; use most recent saved image |

---

### `webScraper.py` — Puzzle scraper
Scrapes today's puzzle from sudoku.com and saves it as a PNG.

**How it works:**
1. Launches a headless Chrome browser via Selenium.
2. Navigates to the daily Sudoku challenge page.
3. Extracts the puzzle `<canvas>` as a base64-encoded PNG.
4. Saves to `DailySudokuChallenges/sudoku_YYYY-MM-DD.png` (Sydney timezone).

**Dependencies:** `selenium`, `webdriver-manager`, `Pillow`, `pytz`

---

### `sudokuExtractor.py` — Board reader
Uses the trained CNN to convert a puzzle image into a 9×9 grid of integers.

**How it works:**
1. Detects the Sudoku grid using adaptive thresholding and contour detection.
2. Divides the grid into 81 individual cell images.
3. Classifies each cell as empty or a digit (1–9) using the CNN.
4. Returns a 9×9 list where `0` represents an empty cell.

```bash
python sudokuExtractor.py [--image PATH] [--model PATH]
```

**Dependencies:** `opencv-python`, `tensorflow`, `numpy`

---

### `sudokuSolver.py` — Backtracking solver
Solves a board extracted from an image and prints the result to the terminal.

**How it works:**
- Recursive backtracking: tries digits 1–9 in each empty cell, backtracks on contradictions.
- Solved cells are marked with `[x]` in the printed output to distinguish them from given clues.

```bash
python sudokuSolver.py [--image PATH] [--model PATH]
```

---

### `sudokuVisualiser.py` — Animated solver
Same backtracking algorithm as `sudokuSolver.py`, with a live matplotlib animation showing each placement (green) and backtrack (red) in real time.

```bash
python sudokuVisualiser.py [--image PATH] [--model PATH] [--speed slow|medium|fast|instant]
```

**Dependencies:** `matplotlib`

---

### `dataCleaning.py` — Training data preprocessor
Loads and preprocesses digit images from `DigitData/` for model training.

**How it works:**
1. Scans `Sample00X/` folders — each folder corresponds to a digit (1–9).
2. Converts images to **greyscale**, resizes to **28×28 px**, inverts (white digit on black), and normalises to `[0, 1]`.
3. Encodes labels as **one-hot vectors** across 9 classes.
4. Splits into **train / validation / test** sets (~72% / 13% / 15%).

**Dependencies:** `opencv-python`, `numpy`, `tensorflow`, `scikit-learn`

---

### `modelTraining.py` — CNN trainer
Builds and trains the digit classifier.

**Model architecture:**

| Layer | Details |
|---|---|
| Input | 28×28×1 greyscale |
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Flatten | — |
| Dropout | 50% |
| Dense (output) | 9 units, Softmax |

- Loss: categorical cross-entropy | Optimiser: Adam
- Saves to `models/model.keras` (timestamped if a file already exists)

```bash
python modelTraining.py [--epochs 20] [--batch_size 64] [--model_save_fpath models/my_model.keras]
```

**Dependencies:** `tensorflow`, `numpy`

---

## Notes

- Chrome must be installed for the web scraper to work.
- The `DigitData/` folder must contain `Sample002` through `Sample010`, each holding PNG images of digits 1–9 respectively.
- The scraper and pipeline use **Australia/Sydney** timezone to determine today's date.
- The web scraper is intended for personal/educational use. Please respect sudoku.com's terms of service.
