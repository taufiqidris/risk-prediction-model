import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    api = KaggleApi()
    api.authenticate()

    os.makedirs("data/raw", exist_ok=True)

    api.dataset_download_files(
        "fdemoribajolin/death-classification-icu",
        path="data/raw",
        unzip=True
    )

if __name__ == "__main__":
    download_dataset()