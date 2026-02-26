# Sudoku Solver

An automated pipeline that scrapes the daily Sudoku puzzle from the web, processes digit images for training data, and trains a Convolutional Neural Network (CNN) to recognise handwritten/printed digits (1–9) for use in a Sudoku solver.

---

## Project Structure

```
SudokuSolver/
├── webScraper.py       # Captures the daily Sudoku puzzle as a PNG screenshot
├── dataCleaning.py     # Loads and preprocesses digit training images
├── modelTraining.py    # Builds, trains, and saves the CNN digit classifier
├── DigitData/          # Training images organised by digit (Sample001–Sample009)
├── DailySudokuChallenges/  # Output folder for scraped puzzle screenshots
└── models/             # Output folder for saved trained models
```

---

## Files

### `webScraper.py`
Scrapes the daily Sudoku challenge from [sudoku.com](https://sudoku.com/challenges/daily-sudoku) and saves it as a PNG image.

**How it works:**
1. Launches a headless Chrome browser using Selenium and ChromeDriverManager.
2. Navigates to the daily Sudoku challenge page.
3. Executes JavaScript to extract the puzzle's `<canvas>` element as a base64-encoded PNG.
4. Decodes the image and saves it to `DailySudokuChallenges/sudoku_YYYY-MM-DD.png`, where the date is based on the **Australia/Sydney** timezone.

**Dependencies:** `selenium`, `webdriver-manager`, `Pillow`, `pytz`

---

### `dataCleaning.py`
Loads and preprocesses the digit training images stored in `DigitData/` ready for model training.

**How it works:**
1. Scans all `Sample00X/` folders inside `DigitData/` — each folder corresponds to a digit (1–9).
2. Reads every PNG image using OpenCV.
3. Converts each image to **greyscale** and resizes it to **28×28 pixels** (the standard input size for digit classifiers).
4. **Inverts** the images so digit strokes are white on a black background (consistent with training conventions).
5. **Normalises** pixel values to the `[0, 1]` range.
6. Converts labels to **one-hot encoding** across 9 classes (digits 1–9; 0 is excluded as it is not a valid Sudoku entry).
7. Splits data into **train (≈72%) / validation (≈13%) / test (15%)** sets with shuffling.

**Dependencies:** `opencv-python`, `numpy`, `tensorflow`, `scikit-learn`

---

### `modelTraining.py`
Defines, trains, and saves a CNN model to classify Sudoku digits (1–9).

**Model architecture (CNN):**
| Layer | Details |
|---|---|
| Input | 28×28×1 greyscale image |
| Conv2D | 32 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 |
| Flatten | — |
| Dropout | 50% (reduces overfitting) |
| Dense (output) | 9 units, Softmax (one per digit 1–9) |

**Training:**
- Loss: Categorical cross-entropy
- Optimiser: Adam
- Default: 10 epochs, batch size 128

**How it works:**
1. Calls `dataCleaning.py` to load and preprocess the training data.
2. Builds and compiles the CNN.
3. Trains on the training set, evaluating against the validation set each epoch.
4. Saves the trained model to `models/model.keras`. If a model already exists at that path, it appends a timestamp to avoid overwriting (e.g. `model_26_02_2026_14_30_00.keras`).

**Command-line usage:**
```bash
python modelTraining.py --epochs 20 --batch_size 64 --model_save_fpath models/my_model.keras
```

**Dependencies:** `tensorflow`, `numpy`

---

## Pipeline Overview

```
DigitData/              webScraper.py
(training images)       (daily puzzle PNG)
       |                       |
dataCleaning.py                |
(preprocess & split)           |
       |                       |
modelTraining.py               |
(train CNN)                    |
       |                       |
models/model.keras   <-->  [future: puzzle solver]
```

---

## Requirements

Install all dependencies with:

```bash
pip install tensorflow opencv-python scikit-learn selenium webdriver-manager Pillow pytz numpy
```

---

## Notes

- The `DigitData/` folder must contain subdirectories `Sample001` through `Sample009`, each holding PNG images of digits 1–9 respectively.
- The `DailySudokuChallenges/` folder must exist before running `webScraper.py`.
- Chrome must be installed on the system for the web scraper to work.
