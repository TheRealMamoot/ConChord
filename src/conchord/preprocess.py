import logging
from pathlib import Path

from librosa.feature import chroma_cqt
from librosa import load, frames_to_time
from mido.midifiles.meta import KeySignatureError
import numpy as np
from pretty_midi import PrettyMIDI
import shutil
from sklearn.preprocessing import normalize
from tqdm import tqdm

from config.config import Config
from utils.logger import setup_logging
from utils.parser import get_preprocess_parser
from utils.utils import load_lab_file, align_labels_to_frames, convert_arff_to_lab

BASE_DIR = Path(__file__).resolve().parents[2] / 'data' / 'processed'
config = Config()
FRAME_DURATION = config.AUDIO_PARAMS['hop_length'] / config.AUDIO_PARAMS['sample_rate']

########## preprocessing ##########
def _generate_lab_files_from_arffs(arff_files: list[str], src_dir: Path, temp_dir: Path) -> None:
    """
    Converts all .arff files into .lab files and saves them into a temporary directory.
    """
    for arff_file in arff_files:
        arff_path = src_dir / arff_file
        lab_path: Path = temp_dir / arff_file.replace('.arff', '.lab')
        if lab_path.exists():
            logging.info(f'Skipping {arff_file} – already processed.')
            continue
        convert_arff_to_lab(arff_path, lab_path)

def save_npz(output_path: Path, **arrays):
    np.savez_compressed(output_path, **arrays)
    logging.info(f'Saved filtered dataset to {output_path}')

def _preprocess_idmt_dataset(dataset: dict, src_dir: Path, output_path: Path) -> None:
    """
    Processes the IDMT dataset by:
    - Loading .wav audio files and corresponding .lab annotation files.
    - Extracting chroma features using Constant-Q Transform (CQT).
    - Aligning chroma frames with chord labels.
    - Saving the resulting data (X, Y_chords, source, category) into a compressed .npz file.
    """
    X, Y_chords, sources, categories = [], [], [], []
    sub_dirs = dataset.get('subdirs')
    for dir in sub_dirs:
        sub_dir: Path = src_dir / dir
        wav_files = [f for f in sub_dir.iterdir() if f.suffix == '.wav']
        annotations = sub_dir / f'{dir}_annotation.lab'
        lab_segments = load_lab_file(annotations)

        for wav_file in tqdm(wav_files, desc=f'IDMT-{dir}', unit='file', ncols=80,
                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} @ {rate_fmt}]'):
            audio, sr = load(wav_file, sr=config.AUDIO_PARAMS['sample_rate'])
            chroma = chroma_cqt(y=audio, sr=sr, hop_length=config.AUDIO_PARAMS['hop_length']).T
            frame_times = frames_to_time(range(chroma.shape[0]), sr=sr, hop_length=config.AUDIO_PARAMS['hop_length'])
            chord_labels = align_labels_to_frames(frame_times, lab_segments)

            X.extend(chroma)
            Y_chords.extend(chord_labels)
            sources.extend(['wav'] * len(chord_labels))
            categories.extend([dir] * len(chord_labels))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output_path,
              X=np.array(X),
              Y_chords=np.array(Y_chords),
              sources=np.array(sources),
              categories=np.array(categories))

def _preprocess_aam_dataset(src_dir: Path, output_path: Path) -> None:
    """
    Processes the AAM dataset by:
    - Converting .arff metadata files into .lab chord annotation files.
    - Loading MIDI files and extracting chroma features from note velocities.
    - Creating binary note presence matrices across 128 MIDI pitches.
    - Aligning chroma frames with chord labels and saving the result
      (X, Y_chords, Y_notes, sources, categories) into a .npz file.
    """
    X, Y_chords, Y_notes, sources, categories = [], [], [], [], []
    temp_dir = BASE_DIR / '_temp_AAM'
    temp_dir.mkdir(parents=True, exist_ok=True)

    mid_files = sorted([f for f in src_dir.iterdir() if f.suffix == '.mid' and 'Drums' not in f.name and 'Demo' not in f.name])
    arff_files = sorted([f.name for f in src_dir.iterdir() if f.name.endswith('beatinfo.arff')])
    _generate_lab_files_from_arffs(arff_files, src_dir, temp_dir)
    logging.info('.lab files created.')

    for midi in tqdm(mid_files, desc='AAM', unit='file', ncols=80,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} @ {rate_fmt}]'):
        midi_id = midi.name[:4]

        try:
            midi_data = PrettyMIDI(str(midi)) # turn Path object to path as a string
        except KeySignatureError:
            tqdm.write(f'[Warning] - Invalid key signature in {midi}, skipping.')
            continue

        lab_path = temp_dir / f'{midi_id}_beatinfo.lab'
        if not lab_path.exists():
            tqdm.write(f'[Warning] Missing .lab file for {midi_id}, skipping.')
            continue

        lab_segments = load_lab_file(lab_path)
        midi_name = midi.name[5:].replace('.mid', '')

        end_time = midi_data.get_end_time()
        frame_times = np.arange(0, end_time, FRAME_DURATION)
        chroma = np.zeros((len(frame_times), 12))
        note_labels = np.zeros((len(frame_times), 128), dtype=np.float32)

        for note in midi_data.instruments[0].notes:
            start_idx = np.searchsorted(frame_times, note.start)
            end_idx = np.searchsorted(frame_times, note.end)
            pitch_class = note.pitch % 12
            velocity = note.velocity
            chroma[start_idx:end_idx, pitch_class] += velocity
            note_labels[start_idx:end_idx, note.pitch] = 1.0

        chroma = normalize(chroma, norm='l1', axis=1)
        chord_labels = align_labels_to_frames(frame_times, lab_segments)

        X.extend(chroma)
        Y_chords.extend(chord_labels)
        Y_notes.extend(note_labels)
        sources.extend(['midi'] * len(chord_labels))
        categories.extend([midi_name] * len(chord_labels))

    shutil.rmtree(temp_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output_path,
              X=np.array(X),
              Y_chords=np.array(Y_chords),
              Y_notes=np.array(Y_notes),
              sources=np.array(sources),
              categories=np.array(categories))
    
def _preprocess_maestro_dataset(dataset: dict, src_dir: Path, output_path: Path) -> None:
    """
    Processes the MAESTRO dataset by:
    - Loading MIDI files and extracting chroma features from note velocities.
    - Creating binary note presence matrices across 128 MIDI pitches and saving the result
      (X, Y_notes, sources, categories) into a .npz file.
    """
    X, Y_notes, sources, categories = [], [], [], []
    sub_dirs = dataset.get('subdirs')

    for dir in sub_dirs:
        sub_dir: Path = src_dir / dir
        mid_files = sorted([f for f in sub_dir.iterdir() if f.suffix == '.midi']) # midi insteas of mid for AAM

        for midi in tqdm(mid_files, desc=f'MAESTRO-{dir}', unit='file', ncols=80,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} @ {rate_fmt}]'):
            try:
                midi_data = PrettyMIDI(str(midi)) # turn Path object to path as a string
            except KeySignatureError:
                tqdm.write(f'[Warning] - Invalid key signature in {midi}, skipping.')
                continue

            end_time = midi_data.get_end_time()
            frame_times = np.arange(0, end_time, FRAME_DURATION)
            chroma = np.zeros((len(frame_times), 12))
            note_labels = np.zeros((len(frame_times), 128), dtype=np.float32)

            for note in midi_data.instruments[0].notes:
                start_idx = np.searchsorted(frame_times, note.start)
                end_idx = np.searchsorted(frame_times, note.end)
                pitch_class = note.pitch % 12
                velocity = note.velocity
                chroma[start_idx:end_idx, pitch_class] += velocity
                note_labels[start_idx:end_idx, note.pitch] = 1.0

            chroma = normalize(chroma, norm='l1', axis=1)

            X.extend(chroma)
            Y_notes.extend(note_labels)
            sources.extend(['midi'] * len(note_labels))
            categories.extend(['piano'] * len(note_labels))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output_path,
              X=np.array(X),
              Y_notes=np.array(Y_notes),
              sources=np.array(sources),
              categories=np.array(categories))

def preprocess(dataset_names: list[str]) -> None:
    """
    Main entry point for preprocessing datasets.
    Iterates through the provided dataset names and triggers the appropriate
    processing function depending on the dataset type.
    Skips processing if the output .npz file already exists.
    """
    for name in dataset_names:
        src = Path(__file__).resolve().parents[2] / 'data' / 'datasets' / name 
        out = BASE_DIR / f'{name}.npz'
        if not src.exists():
            logging.error(f'{src} not found!')
            return
        if out.exists():
            logging.info(f'Skipping {name} – already processed.')
            continue

        logging.info(f'{name} preprocessing started.')
        if name == 'IDMT':
            _preprocess_idmt_dataset(config.DATASETS[name], src, out)
        elif name == 'AAM':
            try:
                _preprocess_aam_dataset(src, out)
            except KeyboardInterrupt:
                logging.warning('Interrupted. Cleaning up...')
                shutil.rmtree(BASE_DIR / '_temp_AAM', ignore_errors=True)
                raise
        elif name == 'MAESTRO':
            _preprocess_maestro_dataset(config.DATASETS[name], src, out)

    logging.info('Preprocessing finished.')

########## filtering data prep ##########
def _validate_ratios(dataset_split_ratios: dict, IDMT_guitar_ratio: float) -> bool:
    if not 0 <= IDMT_guitar_ratio <= 1:
        logging.error('Process Incomplete - Guitar ratios must be between 0 and 1.')
        return False

    if not 0 < sum(dataset_split_ratios.values()) <= 1:
        logging.error('Process Incomplete - Ratios total must be greater than 0 and cannot exceed 1.')
        return False
    
    for name, ratio in dataset_split_ratios.items():
        if not 0 <= ratio <= 1:
            logging.error(f'Process Incomplete - Invalid ratio for {name}:{ratio}.')
    return True

def _validate_instruments(AAM_instruments: list[str]) -> bool:
    target_instruments = set(AAM_instruments)
    valid_instruments = set(config.INSTRUMENTS['AAM'])
    invalid = list(target_instruments - valid_instruments)
    if invalid:
        logging.error(f'Process Incomplete - Invalid instrument(s) for AAM dataset: {invalid}')
        return False
    return True

def _load_dataset(dataset_name: str) -> dict:
    path = Path(BASE_DIR) / f'{dataset_name}.npz'
    if not path.exists():
        logging.error(f'{dataset_name} not found at {path}')
        return {}
    return dict(np.load(path, allow_pickle=True))

def _filter_idmt_dataset(data: dict, IDMT_size: int, guitar_ratio: float, output_path: Path, use_max_size: bool = False) -> None:
    """
    Create a filtered subset of the IDMT dataset or copy full dataset if use_max_size is True.
    """
    if use_max_size:
        source_path = Path(BASE_DIR) / 'IDMT.npz'
        shutil.copy(source_path, output_path)
        logging.info(f'Copied full IDMT dataset to {output_path}')
        return

    X = data['X']
    Y_chords = data['Y_chords']
    categories = data['categories'].astype(str)
    sources = data['sources'].astype(str)

    guitar_mask = categories == 'guitar'
    non_guitar_mask = categories == 'non_guitar'

    guitar_size = int(IDMT_size * guitar_ratio)
    nonguitar_size = IDMT_size - guitar_size

    if guitar_size > np.sum(guitar_mask) or nonguitar_size > np.sum(non_guitar_mask):
        raise ValueError(
            f'Invalid ratios/samples: requested {guitar_size} guitar (max: {np.sum(guitar_mask)}), '
            f'{nonguitar_size} non-guitar (max: {np.sum(non_guitar_mask)}).'
        )

    rng = np.random.default_rng(config.SEED)
    guitar_indices = rng.choice(np.where(guitar_mask)[0], size=guitar_size, replace=False)
    nonguitar_indices = rng.choice(np.where(non_guitar_mask)[0], size=nonguitar_size, replace=False)
    selected_indices = np.concatenate([guitar_indices, nonguitar_indices])
    rng.shuffle(selected_indices)

    save_npz(output_path,
              X=X[selected_indices],
              Y_chords=Y_chords[selected_indices],
              sources=sources[selected_indices],
              categories=categories[selected_indices])
    
def _filter_aam_dataset(data: dict, 
                        AAM_size: int, 
                        AAM_instruments: list[str], 
                        output_path: Path, 
                        use_max_size: bool = False,
                        use_all_aam_instruments: bool = False) -> None:
    """
    Create a filtered subset of AAM with specified instruments.
    Slices entire subsets of AAM for each instrument if use_max_size is True
    """
    X = data['X']
    Y_chords = data['Y_chords']
    Y_notes = data['Y_notes']
    categories = data['categories'].astype(str)
    sources = data['sources'].astype(str)

    rng = np.random.default_rng(config.SEED)
    selected_indices = []

    if use_all_aam_instruments:
        selected_indices = np.arange(len(X))
        if not use_max_size:
            if AAM_size > len(selected_indices):
                raise ValueError(f"Requested {AAM_size} samples, but only {len(selected_indices)} available.")
            selected_indices = rng.choice(selected_indices, size=AAM_size, replace=False)
        rng.shuffle(selected_indices)

    else:
        selected_indices = []
        for inst in AAM_instruments:
            mask = categories == inst
            indices = np.where(mask)[0]

            if not use_max_size:
                available = len(indices)
                samples_per_instrument = AAM_size // len(AAM_instruments)
                if samples_per_instrument > available:
                    raise ValueError(f'Requested {samples_per_instrument} samples for {inst}, but only {available} available.')
                indices = rng.choice(indices, size=samples_per_instrument, replace=False)

            selected_indices.extend(indices)
        rng.shuffle(selected_indices)

    save_npz(output_path,
              X=X[selected_indices],
              Y_chords=Y_chords[selected_indices],
              Y_notes=Y_notes[selected_indices],
              sources=sources[selected_indices],
              categories=categories[selected_indices])
    
def _filter_maestro_dataset(data: dict, MAESTRO_size: int, output_path: Path, use_max_size: bool = False) -> None:
    """
    Create a filtered subset of MAESTRO or copy full dataset if use_max_size is True
    """
    if use_max_size:
        source_path = BASE_DIR / 'MAESTRO.npz'
        shutil.copy(source_path, output_path)
        logging.info(f'Copied full MAESTRO dataset to {output_path}')
        return
    
    X = data['X']
    Y_notes = data['Y_notes']
    categories = data['categories'].astype(str)
    sources = data['sources'].astype(str)

    rng = np.random.default_rng(config.SEED)
    indices = np.arange(len(X))
    selected_indices = rng.choice(indices, size=MAESTRO_size, replace=False)
    rng.shuffle(selected_indices)

    save_npz(output_path,
              X=X[selected_indices],
              Y_notes=Y_notes[selected_indices],
              sources=sources[selected_indices],
              categories=categories[selected_indices])

def filter_data(dataset_names: list[str],
                        filter_size: int,
                        IDMT_ratio: float,
                        AAM_ratio: float,
                        MAESTRO_ratio: float,
                        IDMT_guitar_ratio: float,
                        AAM_instruments: list[str],
                        use_max_size: bool = False,
                        use_all_aam_instruments: bool = False):

    (Path(__file__).resolve().parents[2] / 'data' / 'filtered').mkdir(parents=True, exist_ok=True)

    dataset_split_ratios = {
        'IDMT_ratio':IDMT_ratio,
        'AAM_ratio':AAM_ratio,
        'MAESTRO_ratio':MAESTRO_ratio
    }
    if not _validate_ratios(dataset_split_ratios, IDMT_guitar_ratio):
        return
    if not _validate_instruments(AAM_instruments):
        return

    IDMT_size, AAM_size, MAESTRO_size = [
        int(np.floor(filter_size * ratio)) for ratio in dataset_split_ratios.values()
    ]

    for dataset_name in dataset_names:
        output_path = Path(__file__).resolve().parents[2] / 'data' / 'filtered' / f'{dataset_name}.npz'
        data = _load_dataset(dataset_name)
        if not data:
            return
        
        if dataset_name == 'IDMT':
            _filter_idmt_dataset(data, IDMT_size, IDMT_guitar_ratio, output_path, use_max_size)
        elif dataset_name == 'AAM':
            _filter_aam_dataset(data, AAM_size, AAM_instruments, output_path, use_max_size, use_all_aam_instruments)
        elif dataset_name == 'MAESTRO':
            _filter_maestro_dataset(data, MAESTRO_size, output_path, use_max_size)

    logging.info('Data filtering/slicing complete.')

########## stacking datasets, test-train splits ##########
def stack_and_split_datasets(datasets: list[str] = ['IDMT','AAM','MAESTRO']) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    datasets_dir = Path(__file__).resolve().parents[2] / 'data' / 'filtered'
    all_X, all_Y_chords, all_Y_notes, all_sources = [], [], [], []

    for dataset in datasets:
        ds_path = datasets_dir / f'{dataset}.npz'
        data = np.load(ds_path, allow_pickle=True)
        X = data['X']
        all_X.append(X)
        all_sources.extend(data['sources'])

        if 'Y_notes' in data:
            all_Y_notes.append(data['Y_notes'])
        else:
            all_Y_notes.append(np.full((len(X), 128), np.nan))

        if 'Y_chords' in data:
            all_Y_chords.append(data['Y_chords'])
        else:
            all_Y_chords.append(np.full((len(X),), 'MISSING'))

    X = np.vstack(all_X)
    Y_notes = np.vstack(all_Y_notes)
    Y_chords = np.array(all_Y_chords)
    sources = np.array(all_sources)

    chord_mask = Y_chords != 'MISSING'
    note_mask = ~np.isnan(Y_notes).all(axis=1)

    X_chords = X[chord_mask]
    Y_chords_filtered = Y_chords[chord_mask]
    sources_chords = sources[chord_mask]
    X_notes = X[note_mask]
    Y_notes_filtered = Y_notes[note_mask]
    sources_notes = sources[note_mask]

    splited_data = {
        "chords": (X_chords, Y_chords_filtered, sources_chords),
        "notes": (X_notes, Y_notes_filtered, sources_notes)
    }

    return splited_data

def main():
    setup_logging()
    parser = get_preprocess_parser()
    args = parser.parse_args()
    preprocess(dataset_names=args.datasets)
    filter_data(
        dataset_names=args.datasets,
        filter_size=args.filter_size,
        use_max_size=args.use_max_size,
        use_all_aam_instruments=args.use_all_aam_instruments,
        IDMT_ratio=args.idmt_ratio,
        AAM_ratio=args.aam_ratio,
        MAESTRO_ratio=args.maestro_ratio,
        IDMT_guitar_ratio=args.idmt_guitar_ratio,
        AAM_instruments=args.aam_instruments
    )

if __name__ == '__main__':
    main()    