import argparse

def get_downloader_parser():
    parser = argparse.ArgumentParser(description='Downloader')
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',      
        default=['IDMT','AAM'],
        help='List of dataset names to downlaod (e.g., IDMT)'
    )
    parser.add_argument(
        '--chunksize',
        type=int,
        default=256, # KB  
        help='Chunk size (in MB) for downloading data'
    )
    return parser

def get_preprocess_parser():
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',      
        default=['IDMT','AAM'],
        help='List of dataset names to preprocess (e.g., IDMT)'
    )
    return parser