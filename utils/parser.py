import argparse

def get_base_parser():
    parser = argparse.ArgumentParser(description='ConChord CLI Interface')
    return parser

def get_downloader_parser():
    parser = get_base_parser()
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',      
        default=['IDMT'],
        help='List of dataset names to preprocess (e.g., IDMT)'
    )
    return parser