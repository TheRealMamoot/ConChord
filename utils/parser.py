import argparse

from config.config import Config

config = Config()
defaults = config.DEFAULTS

def get_downloader_parser():
    parser = argparse.ArgumentParser(description='Downloader')
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',      
        default=defaults['downloader']['datasets'],
        choices=defaults['downloader']['datasets'],
        help='List of dataset names to downlaod (e.g., IDMT)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=defaults['downloader']['chunk-size'], # KB  
        help='Chunk size (in KB) for downloading data'
    )
    return parser

def get_preprocess_parser():
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',      
        default=defaults['preprocess']['datasets'],
        choices=defaults['preprocess']['datasets'],
        help='List of dataset names to preprocess (e.g., IDMT)'
    )
    parser.add_argument(
        '--filter-size',
        type=int,
        default=defaults['preprocess']['filter-size'],
        help='Total number of frames to sample (default: 200,000)'
    )
    parser.add_argument(
        '--use-max-size',
        action='store_true',
        help='Use the full available dataset without filtering'
    )
    parser.add_argument(
        '--use-all-aam-instruments',
        action='store_true',
        help='Use all available instruments in AAM dataset'
    )
    parser.add_argument(
        '--idmt-ratio',
        type=float,
        default=defaults['preprocess']['idmt-ratio'],
        help='Portion of total samples to use from IDMT (e.g. 0.4 = 40%)'
    )
    parser.add_argument(
        '--aam-ratio',
        type=float,
        default=defaults['preprocess']['aam-ratio'],
        help='Portion of total samples to use from AAM (e.g. 0.4 = 40%)'
    )
    parser.add_argument(
        '--maestro-ratio',
        type=float,
        default=defaults['preprocess']['maestro-ratio'],
        help='Portion of total samples to use from MAESTRO (e.g. 0.4 = 40%)'
    )
    parser.add_argument(
        '--idmt-guitar-ratio',
        type=float,
        default=defaults['preprocess']['idmt-guitar-ratio'],
        help='Ratio of guitar vs non-guitar samples in IDMT'
    )
    parser.add_argument(
        '--aam-instruments',
        nargs='+',
        default=defaults['preprocess']['aam-instruments'],
        help='List of instruments to include from AAM'
    )
    return parser
