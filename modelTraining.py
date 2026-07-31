from tensorflow import keras
from tensorflow.keras import layers
import json
import os
from datetime import datetime
import argparse

from dataCleaning import FontImage

class DigitClassifier:
    def __init__(self, batch_size=128, epochs=10, model_save_fpath="models/model.keras", exclude_fonts=False):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_save_fpath = model_save_fpath
        self.exclude_fonts = exclude_fonts
        self.model = None

    def buildModel(self):
        """Define and compile the CNN model with training-time augmentation."""
        self.model = keras.Sequential([
            keras.Input(shape=(28, 28, 1)),
            # Augmentation layers — active only during training
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.2),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(9, activation="softmax")
        ])

        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])  


    def loadData(self):
        fontImage = FontImage()
        fontImage.getImageDict()
        return fontImage.getFontArrays()
    
    def trainModel(self, x_train, y_train, x_val, y_val):
        """Train the CNN model."""
        print("Starting training...")
        self.model.fit(x_train, y_train,
                       validation_data=(x_val, y_val),
                       batch_size=self.batch_size,
                       epochs=self.epochs)
        print("Training complete")
    
    def evaluateModel(self, x_test, y_test):
        """Evaluate on hold-out test set and save metrics to results/."""
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Test loss:     {loss:.4f}")
        print(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        os.makedirs("results", exist_ok=True)
        results = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "test_loss": round(float(loss), 6),
            "test_accuracy": round(float(accuracy), 6),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }
        path = os.path.join("results", "training_results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {path}")
        return loss, accuracy

    def saveModel(self):
        """Save the trained model to a file."""
        os.makedirs(os.path.dirname(self.model_save_fpath), exist_ok=True)  # Ensure directory exists
        
        if os.path.exists(self.model_save_fpath):
            now = datetime.now()
            suffix = now.strftime("%d_%m_%Y_%H_%M_%S")
            self.model_save_fpath = f"models/model_{suffix}.keras"

        self.model.save(self.model_save_fpath)
        print(f"Model saved at: {self.model_save_fpath}")

    
    def run(self):
        """Execute the full pipeline."""
        # Load data
        x_train, x_val, x_test, y_train, y_val, y_test = self.loadData()

        # Build and compile the model
        self.buildModel()

        # Train the model
        self.trainModel(x_train, y_train, x_val, y_val)

        # Evaluate on test set and save results
        self.evaluateModel(x_test, y_test)

        # Save the model
        self.saveModel()

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_save_fpath", default="models/model.keras", type=str)
    ap.add_argument("--batch_size", default=128, type=int)
    ap.add_argument("--epochs", default=10, type=int)

    args = vars(ap.parse_args())

    # Create an instance of the DigitClassifier and run it
    classifier = DigitClassifier(
        batch_size=args['batch_size'],
        epochs=args['epochs'],
        model_save_fpath=args['model_save_fpath']
    )
    classifier.run()

