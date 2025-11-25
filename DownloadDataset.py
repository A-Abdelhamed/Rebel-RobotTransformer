import os
import kaggle

# 1. MANUALLY SET CREDENTIALS HERE
os.environ['KAGGLE_USERNAME'] = "ahmedkzabdelhamed"
os.environ['KAGGLE_KEY'] = "d337c88281ad430b9f4868decbb4178f"

# 2. Authenticate
kaggle.api.authenticate()

# 3. Download
print("Starting download...")
kaggle.api.dataset_download_files('ahmedkzabdelhamed/igus-tool-picking', path='.', unzip=True)
print("Success!")