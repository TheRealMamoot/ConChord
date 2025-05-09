import logging
import os
import shutil
import zipfile

import requests
from tqdm import tqdm

from utils.logger import setup_logging
from utils.parser import get_downloader_parser
from utils.utils import folder_is_populated
from config.config import DATASETS

BASE_DIR = 'data/datasets'

def download_and_extract(dataset_names: list[str] = ['IDMT','AMM'], chunk_size: int = 256): # in MB

    os.makedirs(BASE_DIR, exist_ok=True)

    try:
        for dataset_name in dataset_names:
            if dataset_name not in DATASETS:
                logging.error(f'Dataset "{dataset_name}" not found in config.')
                raise FileNotFoundError(f'Required dataset "{dataset_name}" not found.')

            dataset: dict = DATASETS[dataset_name]
            zip_path = os.path.join(BASE_DIR, f'{dataset_name}.zip')
            extract_temp = os.path.join(BASE_DIR, f'_temp_{dataset_name}')
            final_dir = os.path.join(BASE_DIR, dataset_name)

            # Only skip if all expected subdirs are present and non-empty
            skip = True
            subdirs = dataset.get('subdirs')

            if subdirs is not None:
                for subdir in subdirs:
                    full_path = os.path.join(final_dir, subdir)
                    if not os.path.exists(full_path) or not folder_is_populated(full_path):
                        skip = False
                        break
                if skip:
                    logging.info(f'{dataset_name} is already fully extracted and populated at {final_dir}.')
                    continue
            else:
                if os.path.exists(final_dir) and os.listdir(final_dir):
                    logging.info(f'{dataset_name} already exists and is non-empty at {final_dir}.')
                    continue

            # Download and extract
            if os.path.exists(extract_temp):
                shutil.rmtree(extract_temp)

            if not os.path.exists(zip_path):
                urls = dataset.get('url')
                for url in urls:
                    r = requests.get(url, stream=True)
                    total = int(r.headers.get('content-length', 0))
                    with open(zip_path, 'wb') as f, tqdm(
                        total=total,
                        desc=f'{dataset_name}',
                        unit='B',
                        unit_scale=True,
                        ncols=80,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} @ {rate_fmt}'
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=chunk_size * 1024):
                            f.write(chunk)
                            bar.update(len(chunk))
                        try:
                            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                zip_ref.extractall(extract_temp)
                        except zipfile.BadZipFile:
                            logging.error(f'Corrupted zip file: {zip_path}. Deleting it...')
                            os.remove(zip_path)
                            raise RuntimeError(
                                f'The zip file for {dataset_name} was corrupted or incomplete. '
                                f'Please rerun the downloader to fetch it again.'
                            )
                logging.info(f'{dataset_name} download complete!')

            # Prepare final destination (lol)
            if os.path.exists(final_dir):
                for item in os.listdir(final_dir):
                    item_path = os.path.join(final_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    elif item != '.gitkeep':
                        os.remove(item_path)
            else:
                os.makedirs(final_dir, exist_ok=True)

            # Move specified subdirectories
            if subdirs:
                for subdir in subdirs:
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

            os.remove(zip_path)
            shutil.rmtree(extract_temp)
            logging.info(f'{dataset_name} folder created!')

    except KeyboardInterrupt:
        logging.warning('Download interrupted. Cleaning up...')
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise

def main():
    setup_logging()
    parser = get_downloader_parser()
    args = parser.parse_args()
    download_and_extract(dataset_names=args.datasets, chunk_size=args.chunksize)

if __name__ == '__main__':
    main()