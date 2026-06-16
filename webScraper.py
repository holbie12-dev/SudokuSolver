from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from PIL import Image
import base64
import glob
import io
import os
import datetime
import pytz

# Selenium Manager and webdriver-manager both phone home to
# googlechromelabs.github.io to resolve the driver version, which fails when
# offline or behind restrictive DNS. Reuse a chromedriver already cached by
# webdriver-manager (~/.wdm) instead of looking one up over the network.
def _cached_chromedriver_path():
    cache_glob = os.path.join(os.path.expanduser('~'), '.wdm', 'drivers', 'chromedriver', '**', 'chromedriver.exe')
    matches = glob.glob(cache_glob, recursive=True)
    return max(matches, key=os.path.getmtime) if matches else None

class SudokuScraper:
    def __init__(self):
        self.url = 'https://sudoku.com/challenges/daily-sudoku'

    def get_canvas_image(self):
        # Set up the Selenium WebDriver
        driver_path = _cached_chromedriver_path()
        service = Service(executable_path=driver_path) if driver_path else Service()
        driver = webdriver.Chrome(service=service)
        
        # Open the page with the canvas element
        driver.get(self.url)
        
        # Wait for the page to load and get the canvas image as base64 PNG
        canvas_image_base64 = driver.execute_script("""
            var canvas = document.getElementById('game').getElementsByTagName('canvas')[0];
            return canvas.toDataURL('image/png');
        """)
        
        # Extract the base64 image part from the data URL
        image_data = canvas_image_base64.split('base64,')[-1]
        
        # Convert the base64 string to binary data
        img_data = base64.b64decode(image_data)


        # Save the image using Pillow
        image = Image.open(io.BytesIO(img_data))

        # Folder Path
        folderPath = os.path.join(os.getcwd(), 'DailySudokuChallenges')

        #Get Timezone
        timezone = pytz.timezone('Australia/Sydney')
        currentDate = datetime.datetime.now(timezone).strftime('%Y-%m-%d')
        image.save(os.path.join(folderPath, f"sudoku_{currentDate}.png"))  # Save the image as a PNG
        
        # Close the WebDriver
        driver.quit()
        saved_path = os.path.join(folderPath, f"sudoku_{currentDate}.png")
        print(f"Image saved as sudoku_{currentDate}.png.")
        return saved_path


if __name__ == '__main__':
    scraper = SudokuScraper()
    scraper.get_canvas_image()
