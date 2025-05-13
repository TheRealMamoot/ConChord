import logging
import os
from pathlib import Path
import shutil
import zipfile

import requests
from tqdm import tqdm

from utils.logger import setup_logging
from utils.parser import get_downloader_parser
from utils.utils import folder_is_populated
from config.config import DATASETS

BASE_DIR = Path('data') / 'datasets'

def download_and_extract(dataset_names: list[str] = ['IDMT', 'AMM'], chunk_size: int = 256):
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for dataset_name in dataset_names:
            if dataset_name not in DATASETS:
                logging.error(f'Dataset "{dataset_name}" not found in config.')
                raise FileNotFoundError(f'Required dataset "{dataset_name}" not found.')

            dataset = DATASETS[dataset_name]
            zip_path = BASE_DIR / f'{dataset_name}.zip'
            extract_temp = BASE_DIR / f'_temp_{dataset_name}'
            final_dir = BASE_DIR / dataset_name

            skip = True
            subdirs = dataset.get('subdirs')

            if subdirs is not None:
                for subdir in subdirs:
                    full_path = final_dir / subdir
                    if not full_path.exists() or not folder_is_populated(full_path):
                        skip = False
                        break
                if skip:
                    logging.info(f'{dataset_name} is already fully extracted and populated at {final_dir}.')
                    continue
            else:
                if final_dir.exists() and any(final_dir.iterdir()):
                    logging.info(f'{dataset_name} already exists and is non-empty at {final_dir}.')
                    continue

            if extract_temp.exists():
                shutil.rmtree(extract_temp)

            if not zip_path.exists():
                urls = dataset.get('url')
                for url in urls:
                    r = requests.get(url, stream=True)
                    total = int(r.headers.get('content-length', 0))
                    with zip_path.open('wb') as f, tqdm(
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
                    zip_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f'The zip file for {dataset_name} was corrupted or incomplete. '
                        f'Please rerun the downloader to fetch it again.'
                    )
                logging.info(f'{dataset_name} download complete!')

            if final_dir.exists():
                for item in final_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    elif item.name != '.gitkeep':
                        item.unlink()
            else:
                final_dir.mkdir(parents=True, exist_ok=True)

            if subdirs:
                for subdir in subdirs:
                    found = False
                    for root, dirs, _ in os.walk(extract_temp):
                        if subdir in dirs:
                            src = Path(root) / subdir
                            dst = final_dir / subdir
                            logging.info(f'Moving {src} → {dst}')
                            shutil.move(str(src), str(dst))
                            found = True
                            break
                    if not found:
                        logging.warning(f'Could not find subdir "{subdir}" in extracted zip.')
            else:
                for item in extract_temp.iterdir():
                    dst = final_dir / item.name
                    logging.info(f'Moving {item} → {dst}')
                    shutil.move(str(item), str(dst))

            zip_path.unlink(missing_ok=True)
            shutil.rmtree(extract_temp, ignore_errors=True)
            logging.info(f'{dataset_name} folder created!')

    except KeyboardInterrupt:
        logging.warning('Download interrupted. Cleaning up...')
        zip_path.unlink(missing_ok=True)
        raise

def main():
    setup_logging()
    parser = get_downloader_parser()
    args = parser.parse_args()
    download_and_extract(dataset_names=args.datasets, chunk_size=args.chunksize)
if __name__ == '__main__':
    main()