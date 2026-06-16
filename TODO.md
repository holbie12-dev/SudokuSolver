# SudokuSolver — TODO

## Model Quality
- [ ] Expand the training dataset — 9 sample folders is very small; add digit images from MNIST or augment existing samples (rotation, brightness, font variations)
- [ ] Fine-tune from a pre-trained MNIST model rather than training from scratch — faster convergence on limited data
- [ ] Save test accuracy and confusion matrix to a file (`models/metrics.json`) after training — currently only printed

## Pipeline Robustness
- [ ] Add error handling in `sudokuExtractor.py` for misaligned grids or low-confidence digit predictions (fallback to manual entry)
- [ ] Handle puzzles that are already partially solved (given digits pre-filled) — current pipeline assumes a blank grid
- [ ] Cache the trained model by hash of training data so `dailyRunner.py` doesn't retrain when nothing has changed

## Scraper
- [ ] Check whether the Selenium + webdriver-manager setup still works with current sudoku.com markup — last confirmed Jan 2025
- [ ] Add a fallback image source (e.g. local test PNG) so `dailyRunner.py` can run offline for development

## Visualiser
- [ ] Save the completed animation as an MP4 alongside the solved grid image
- [ ] Show the original unsolved grid at the start of the animation for context

## Quality
- [ ] Add a unit test that runs the backtracking solver on a known hard puzzle and checks the solution
- [ ] Write a short `README.md` covering setup, how to run `dailyRunner.py`, and expected output
