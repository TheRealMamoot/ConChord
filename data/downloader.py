import logging
import os
import shutil
import zipfile

import requests
from tqdm import tqdm

from utils.logger import setup_logging

def download_and_extract(url: str, dataset_name: str, extract_subdirs: list[str] | None = None):
    base_dir = 'data/datasets'
    zip_path = os.path.join(base_dir, f'{dataset_name}.zip')
    extract_temp = os.path.join(base_dir, f'_temp_{dataset_name}')
    final_dir = os.path.join(base_dir, dataset_name)
    if os.path.exists(final_dir):
        logging.info(f'{dataset_name} already exists at {final_dir}, skipping download and extraction.')
        return

    os.makedirs(base_dir, exist_ok=True)

    # Download
    if not os.path.exists(zip_path):
        logging.info(f'Downloading {dataset_name}...')
        r = requests.get(url, stream=True)
        total = int(r.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=262144):
                f.write(chunk)
                bar.update(len(chunk))
        logging.info(f'{dataset_name} download complete!')

    # Extract
    if os.path.exists(extract_temp):
        shutil.rmtree(extract_temp)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_temp)

    # Specified subdirectories
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)
    os.makedirs(final_dir, exist_ok=True)

    # Move specified subdirectories
    if extract_subdirs:
        for subdir in extract_subdirs:
            found = False
            for root, dirs, _ in os.walk(extract_temp):
                if subdir in dirs:
                    src = os.path.join(root, subdir)
                    dst = os.path.join(final_dir, subdir)
                    logging.info(f'Moving {src} → {dst}')
                    shutil.move(src, dst)
                    found = True
                    break
            if not found:
                logging.warning(f'Could not find subdir "{subdir}" in extracted zip.')
    else:
        for item in os.listdir(extract_temp):
            src = os.path.join(extract_temp, item)
            dst = os.path.join(final_dir, item)
            logging.info(f'Moving {src} → {dst}')
            shutil.move(src, dst)

    shutil.rmtree(extract_temp)
    os.remove(zip_path)
    logging.info(f'{dataset_name} folder created!')

def main():
    setup_logging()
    download_and_extract(
        url = 'https://zenodo.org/records/7544213/files/IDMT-SMT-CHORDS.zip?download=1',
        dataset_name='IDMT',
        extract_subdirs=['guitar','non_guitar']
    )

if __name__ == '__main__':
    main()