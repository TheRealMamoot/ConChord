import argparse

def get_downloader_parser():
    parser = argparse.ArgumentParser(description='Downloader')
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',      
        default=['IDMT'],
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
        default=['IDMT'],
        help='List of dataset names to preprocess (e.g., IDMT)'
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=11025, # Hz
        help='Sample rate conversion from the original audio.'
    )
    parser.add_argument(
        '--hoplength',
        type=int,
        default=512,
        help='Number of “hops forward” in the audio signal between frames for extracting features (e.g, chromagrams)'
    )
    return parser