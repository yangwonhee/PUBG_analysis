import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract_data(dataset, save_dir, extract_to):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=save_dir, unzip=False)

    with zipfile.ZipFile(os.path.join(save_dir, f"{dataset.split('/')[-1]}.zip"), 'r') as zip_ref:
        zip_ref.extractall(extract_to)