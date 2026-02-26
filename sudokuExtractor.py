import cv2
import numpy as np
import tensorflow as tf
import os
import glob
import argparse


class SudokuExtractor:
    def __init__(self, model_path="models/model.keras"):
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded.")

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def load_image(self, image_path=None, folder="DailySudokuChallenges"):
        """Load a specific image or the most recent one in the folder."""
        if image_path:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")
            return img, image_path

        images = sorted(glob.glob(os.path.join(folder, "*.png")))
        if not images:
            raise FileNotFoundError(f"No PNG images found in '{folder}/'")
        latest = images[-1]
        return cv2.imread(latest), latest

    # ------------------------------------------------------------------
    # Grid detection
    # ------------------------------------------------------------------

    def find_grid(self, image):
        """Locate and return the cropped Sudoku grid from the full image.

        The sudoku.com canvas renders the grid as 9 separate 3x3 box regions.
        Adaptive thresholding therefore finds 9 large contours (one per box)
        rather than one outer rectangle. This method takes the union bounding
        box of all significant contours to recover the full grid extent.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=2
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Could not find any contours — check the input image.")

        # Keep only contours that are at least 3% of the total image area.
        # Each 3x3 sub-box qualifies; noise and small artefacts do not.
        min_area = image.shape[0] * image.shape[1] * 0.03
        large = [c for c in contours if cv2.contourArea(c) > min_area]
        if not large:
            raise ValueError("No large contours found — grid may not be visible.")

        # Compute the union bounding box across all qualifying contours.
        x_min = min(cv2.boundingRect(c)[0] for c in large)
        y_min = min(cv2.boundingRect(c)[1] for c in large)
        x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in large)
        y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in large)

        return image[y_min:y_max, x_min:x_max]

    # ------------------------------------------------------------------
    # Cell extraction
    # ------------------------------------------------------------------

    def extract_cells(self, grid):
        """Divide the grid into 81 individual cell images (row-major order)."""
        h, w = grid.shape[:2]
        cell_h = h // 9
        cell_w = w // 9

        cells = []
        for row in range(9):
            for col in range(9):
                y1 = row * cell_h
                y2 = (row + 1) * cell_h
                x1 = col * cell_w
                x2 = (col + 1) * cell_w
                cells.append(grid[y1:y2, x1:x2])

        return cells

    # ------------------------------------------------------------------
    # Cell classification helpers
    # ------------------------------------------------------------------

    def is_empty(self, cell):
        """Return True if the cell contains no digit.

        Looks at the inner 60% of the cell (to exclude grid border lines)
        and counts dark pixels. Uses a threshold proportional to the inner
        area so it works regardless of the source image resolution.
        """
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        pad_y = int(h * 0.2)
        pad_x = int(w * 0.2)
        inner = gray[pad_y:h - pad_y, pad_x:w - pad_x]

        # Digits are dark (< 180 intensity) on a light background
        _, thresh = cv2.threshold(inner, 180, 255, cv2.THRESH_BINARY_INV)
        # Empty if fewer than 0.5% of inner pixels are dark
        return cv2.countNonZero(thresh) < inner.size * 0.005

    def preprocess_for_model(self, cell):
        """Preprocess a cell to match the CNN training format.

        Steps match dataCleaning.py:
          1. Greyscale
          2. Crop inner 80% (removes grid border lines)
          3. Resize to 28x28
          4. Invert (white digit on black background)
          5. Normalise to [0, 1]
          6. Reshape to (1, 28, 28, 1)
        """
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        pad_y = int(h * 0.1)
        pad_x = int(w * 0.1)
        inner = gray[pad_y:h - pad_y, pad_x:w - pad_x]

        resized = cv2.resize(inner, (28, 28), interpolation=cv2.INTER_AREA)
        inverted = cv2.bitwise_not(resized)
        normalised = inverted.astype("float32") / 255

        return normalised.reshape(1, 28, 28, 1)

    def predict_digit(self, cell):
        """Return the predicted digit (1–9) for a non-empty cell."""
        preprocessed = self.preprocess_for_model(cell)
        probabilities = self.model.predict(preprocessed, verbose=0)
        # Model outputs 9 classes (index 0 = digit 1, ..., index 8 = digit 9)
        return int(np.argmax(probabilities)) + 1

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def extract_board(self, image_path=None):
        """Full pipeline: image -> grid crop -> 81 cells -> classified board.

        Returns:
            list[list[int]]: 9x9 board where 0 represents an empty cell.
        """
        image, path = self.load_image(image_path)
        print(f"Processing: {path}")

        grid = self.find_grid(image)
        cells = self.extract_cells(grid)

        board_flat = []
        for cell in cells:
            if self.is_empty(cell):
                board_flat.append(0)
            else:
                board_flat.append(self.predict_digit(cell))

        # Reshape flat list into 9x9
        return [board_flat[i * 9:(i + 1) * 9] for i in range(9)]

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    @staticmethod
    def print_board(board):
        """Pretty-print the board with box dividers."""
        divider = "+-------+-------+-------+"
        print("\nExtracted Sudoku Board:")
        print(divider)
        for i, row in enumerate(board):
            line = "| "
            for j, val in enumerate(row):
                line += (str(val) if val != 0 else ".") + " "
                if (j + 1) % 3 == 0 and j < 8:
                    line += "| "
            line += "|"
            print(line)
            if (i + 1) % 3 == 0:
                print(divider)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract digits from a scraped Sudoku puzzle image.")
    ap.add_argument(
        "--image", default=None, type=str,
        help="Path to a specific puzzle PNG. Defaults to the most recent file in DailySudokuChallenges/."
    )
    ap.add_argument(
        "--model", default="models/model.keras", type=str,
        help="Path to the trained digit classifier model."
    )
    args = ap.parse_args()

    extractor = SudokuExtractor(model_path=args.model)
    board = extractor.extract_board(image_path=args.image)

    SudokuExtractor.print_board(board)

    print("\nBoard as Python list:")
    print(board)
