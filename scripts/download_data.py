import os
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "fdemoribajolin/death-classification-icu"
OUT_DIR = "data/raw"

def main():
    print("Authenticating Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    print("✅ Auth OK")

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Downloading {DATASET} to {OUT_DIR} ...")

    api.dataset_download_files(
        DATASET,
        path=OUT_DIR,
        unzip=True
    )

    print("✅ Download complete. Files in data/raw:")
    print(os.listdir(OUT_DIR))

if __name__ == "__main__":
    main()