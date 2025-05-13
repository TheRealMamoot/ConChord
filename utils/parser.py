import argparse

def get_downloader_parser():
    parser = argparse.ArgumentParser(description='Downloader')
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',      
        default=['IDMT','AAM','MAESTRO'],
        choices=['IDMT','AAM','MAESTRO'],
        help='List of dataset names to downlaod (e.g., IDMT)'
    )
    parser.add_argument(
        '--chunk-size',
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
        choices=['IDMT','AAM'],
        help='List of dataset names to preprocess (e.g., IDMT)'
    )
    parser.add_argument(
        '--filter-size',
        type=int,
        default=200_000,
        help='Total number of frames to sample (default: 200,000)'
    )
    parser.add_argument(
        '--use-max-size',
        action='store_true',
        help='Use the full available dataset without filtering'
    )
    parser.add_argument(
        '--dataset-split-ratio',
        type=float,
        default=0.5,
        help='Ratio of IDMT to AAM in the filtered dataset (e.g, 0.8: 80% IDMT, 20% AAM)'
    )
    parser.add_argument(
        '--IDMT-guitar-ratio',
        type=float,
        default=0.5,
        help='Ratio of guitar vs non-guitar samples in IDMT'
    )
    parser.add_argument(
        '--AAM-instruments',
        nargs='+',
        default=['AcousticGuitar', 'Piano'],
        help='List of instruments to include from AAM'
    )
    return parser
