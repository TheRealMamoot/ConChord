import logging
import shutil
from pathlib import Path

import numpy as np
from librosa import frames_to_time, load
from librosa.feature import chroma_cqt
from mido.midifiles.meta import KeySignatureError
from pretty_midi import PrettyMIDI
from sklearn.preprocessing import normalize
from tqdm import tqdm

from src.conchord.config.config import Config
from src.conchord.config.logger import setup_logger
from src.conchord.utils.parser import get_preprocess_parser
from src.conchord.utils.utils import align_labels_to_frames, convert_arff_to_lab, load_lab_file

config = Config()
PREPROCESS_DIR = Path(__file__).resolve().parents[2] / 'data' / 'preprocessed'
FRAME_DURATION = config.AUDIO_PARAMS['hop_length'] / config.AUDIO_PARAMS['sample_rate']
SEQUENCE_DURATION = 1  # second
SEQUENCE_LENGTH = int(np.ceil(SEQUENCE_DURATION / FRAME_DURATION))  # 22 frames per second


# =================
# === Utilities ===
# =================


def _generate_lab_files_from_arffs(arff_files: list[str], src_dir: Path, temp_dir: Path) -> None:
    """
    Convert all .arff files into .lab files and save them into a temporary directory.
    """
    for arff_file in arff_files:
        arff_path = src_dir / arff_file
        lab_path: Path = temp_dir / arff_file.replace('.arff', '.lab')
        if lab_path.exists():
            logging.info(f'Skipping {arff_file} – already preprocessed.')
            continue
        convert_arff_to_lab(arff_path, lab_path)


def _validate_ratios(dataset_split_ratios: dict, IDMT_guitar_ratio: float | None = None) -> bool:
    if IDMT_guitar_ratio is not None:
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


def _load_dataset_to_filter(dataset_name: str) -> dict:
    path = Path(PREPROCESS_DIR) / f'{dataset_name}.npz'
    if not path.exists():
        logging.error(f'{dataset_name} not found at {path}')
        return {}
    return dict(np.load(path, allow_pickle=True))


def save_npz(output_path: Path, **arrays) -> None:
    np.savez_compressed(output_path, **arrays)
    logging.info(f'Saved subset to {output_path}')


def trim_to_sequence_length(*arrays, seq_len: int) -> tuple[np.ndarray, ...]:
    """
    Trims all input arrays so their length becomes a multiple of seq_len.
    Returns the trimmed arrays in the same order.
    """
    trimmed = []
    for arr in arrays:
        num_sequences = len(np.array(arr)) // seq_len
        cut_off = num_sequences * seq_len
        trimmed.append(arr[:cut_off])
    return tuple(trimmed)


def add_chroma_noise(
    chroma: np.ndarray,
    silence_noise_uniform: tuple[float, float] = (1.0, 5.0),
    general_noise_normal: tuple[float, float] = (0.5, 2.0),
) -> np.ndarray:
    """
    Adds noise to chroma features:
    - Adds random uniform noise to silent bins (originally zero).
    - Adds Gaussian noise across all bins to simulate real-world imperfections.
    Args:
        silence_noise_uniform: Min and max range for Uniform silence noise !after normalization.
        general_noise_normal: Mean and standard deviation of Gaussian noise !after normalization.
    """
    rng = np.random.default_rng(seed=config.SEED)
    chroma = chroma.copy()

    # Random uniform noise for silent bins
    silence_mask = chroma == 0
    chroma[silence_mask] = rng.uniform(
        silence_noise_uniform[0],
        silence_noise_uniform[1],
        size=silence_mask.sum(),
    )

    # Add Gaussian noise to entire matrix
    chroma += rng.normal(loc=general_noise_normal[0], scale=general_noise_normal[1], size=chroma.shape)
    chroma = np.clip(chroma, a_min=0.01, a_max=None)  # No negative chroma values

    return chroma


# =====================
# === Preprocessing ===
# =====================


def _preprocess_idmt_dataset(
    dataset: dict,
    src_dir: Path,
    output_path: Path,
    hop_len: int,
    sample_rate: int,
    seq_len: int = SEQUENCE_LENGTH,
) -> None:
    """
    Processes the IDMT dataset by:
    - Loads .wav audio files and corresponding .lab annotation files.
    - Extracts chroma features using Constant-Q Transform (CQT).
    - Aligns chroma frames with chord labels.
    - Reshapes frames into sequences.
    Args:
        hop_len: Number of audio samples between each chroma frame. Controls temporal resolution.
        sample_rate (Hz): Sampling rate to which the audio should be resampled.
        seq_len: Number of chroma frames per sequence. Defines the temporal length of each training sample.
    """
    X, Y_chords, sources, categories = [], [], [], []
    sub_dirs: list = dataset['subdirs']
    for dir in sub_dirs:
        sub_dir: Path = src_dir / dir
        wav_files = [f for f in sub_dir.iterdir() if f.suffix == '.wav']
        annotations = sub_dir / f'{dir}_annotation.lab'
        lab_segments = load_lab_file(annotations)

        for wav_file in tqdm(
            wav_files,
            desc=f'IDMT-{dir}',
            unit='file',
            ncols=80,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} @ {rate_fmt}]',
        ):
            audio, sr = load(wav_file, sr=sample_rate)
            chroma = chroma_cqt(y=audio, sr=sr, hop_length=hop_len).T
            frame_times = frames_to_time(range(chroma.shape[0]), sr=sr, hop_length=hop_len)
            chord_labels = align_labels_to_frames(frame_times, lab_segments)

            num_sequences = len(chroma) // seq_len
            chroma, chord_labels = trim_to_sequence_length(chroma, chord_labels, seq_len=seq_len)
            chroma = normalize(chroma, norm='l1', axis=1)

            # Reshape into sequences
            chroma = chroma.reshape(num_sequences, seq_len, chroma.shape[1])
            chord_labels = np.array(chord_labels).reshape(num_sequences, seq_len)

            X.extend(chroma)
            Y_chords.extend(chord_labels)
            sources.extend(['wav'] * len(chord_labels))
            categories.extend([dir] * len(chord_labels))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(
        output_path,
        X=np.array(X),
        Y_chords=np.array(Y_chords),
        sources=np.array(sources),
        categories=np.array(categories),
    )


def _preprocess_aam_dataset(
    src_dir: Path,
    output_path: Path,
    silence_noise: tuple[float, float],
    general_noise: tuple[float, float],
    seq_len: int = SEQUENCE_LENGTH,
) -> None:
    """
    Processes the AAM dataset by:
    - Converts .arff metadata files into .lab chord annotation files.
    - Loads MIDI files and extracting chroma features from note velocities.
    - Creates binary note presence matrices across 128 MIDI pitches.
    - Aligns chroma frames with chord labels.
    - Exclude completely silent bins.
    - Adds random noise to chroma bins.
    - Reshapes frames into sequences.
    Args:
        silence_noise: Uniform distribution Z~U(a,b)
        general_noise: Normal distribution Z~N(µ,σ)
    """
    X, Y_chords, Y_notes, sources, categories = [], [], [], [], []
    temp_dir = PREPROCESS_DIR / '_temp_AAM'
    temp_dir.mkdir(parents=True, exist_ok=True)

    mid_files = sorted(
        [f for f in src_dir.iterdir() if f.suffix == '.mid' and 'Drums' not in f.name and 'Demo' not in f.name]
    )
    arff_files = sorted([f.name for f in src_dir.iterdir() if f.name.endswith('beatinfo.arff')])
    _generate_lab_files_from_arffs(arff_files, src_dir, temp_dir)
    logging.info('.lab files created')

    for midi in tqdm(
        mid_files,
        desc='AAM',
        unit='file',
        ncols=80,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} @ {rate_fmt}]',
    ):
        midi_id = midi.name[:4]

        try:
            midi_data = PrettyMIDI(str(midi))  # turn Path object to path as a string``
        except KeySignatureError:
            tqdm.write(f'[Warning] - Invalid key signature in {(midi.name)[:-4]}, skipping')
            continue

        lab_path = temp_dir / f'{midi_id}_beatinfo.lab'
        if not lab_path.exists():
            tqdm.write(f'[Warning] Missing .lab file for {midi_id}, skipping')
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

        # Exclude all zero chromas (absolute silence)
        valid_rows = np.any(chroma, axis=1)
        chroma = chroma[valid_rows]
        note_labels = note_labels[valid_rows]
        frame_times = frame_times[valid_rows]

        chroma = add_chroma_noise(chroma, silence_noise, general_noise)
        chroma = normalize(chroma, norm='l1', axis=1)
        chord_labels = align_labels_to_frames(frame_times, lab_segments)

        num_sequences = len(chroma) // seq_len
        chroma, chord_labels, note_labels = trim_to_sequence_length(chroma, chord_labels, note_labels, seq_len=seq_len)

        chroma = chroma.reshape(num_sequences, seq_len, chroma.shape[1])
        chord_labels = np.array(chord_labels).reshape(num_sequences, seq_len)
        note_labels = np.array(note_labels).reshape(num_sequences, seq_len, 128)

        X.extend(chroma)
        Y_chords.extend(chord_labels)
        Y_notes.extend(note_labels)
        sources.extend(['midi'] * len(chord_labels))
        categories.extend([midi_name] * len(chord_labels))

    shutil.rmtree(temp_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(
        output_path,
        X=np.array(X),
        Y_chords=np.array(Y_chords),
        Y_notes=np.array(Y_notes),
        sources=np.array(sources),
        categories=np.array(categories),
    )


def _preprocess_maestro_dataset(
    dataset: dict,
    src_dir: Path,
    output_path: Path,
    silence_noise: tuple[float, float],
    general_noise: tuple[float, float],
    seq_len: int = SEQUENCE_LENGTH,
) -> None:
    """
    Processes the MAESTRO dataset:
    - Loads MIDI files and extracting chroma features from note velocities.
    - Creates binary note presence matrices across 128 MIDI pitches.
    - Adds random noise to chroma bins.
    - Reshapes frames into sequences.
    """
    X, Y_notes, sources, categories = [], [], [], []
    sub_dirs: list[str] = dataset['subdirs']

    for dir in sub_dirs:
        sub_dir: Path = src_dir / dir
        mid_files = sorted([f for f in sub_dir.iterdir() if f.suffix == '.midi'])  # midi insteas of mid for AAM

        for midi in tqdm(
            mid_files,
            desc=f'MAESTRO-{dir}',
            unit='file',
            ncols=80,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} @ {rate_fmt}]',
        ):
            try:
                midi_data = PrettyMIDI(str(midi))  # turn Path object to path as a string
            except KeySignatureError:
                tqdm.write(f'[Warning] - Invalid key signature in {midi}, skipping')
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

            chroma = add_chroma_noise(chroma, silence_noise, general_noise)
            chroma = normalize(chroma, norm='l1', axis=1)

            chroma, note_labels = trim_to_sequence_length(chroma, note_labels, seq_len=seq_len)
            num_sequences = len(chroma) // seq_len

            chroma = chroma.reshape(num_sequences, seq_len, chroma.shape[1])
            note_labels = np.array(note_labels).reshape(num_sequences, seq_len, 128)

            X.extend(chroma)
            Y_notes.extend(note_labels)
            sources.extend(['midi'] * len(note_labels))
            categories.extend(['piano'] * len(note_labels))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(
        output_path,
        X=np.array(X),
        Y_notes=np.array(Y_notes),
        sources=np.array(sources),
        categories=np.array(categories),
    )


def preprocess_data(
    dataset_names: list[str],
    idmt_hop_len: int,
    idmt_sr: int,
    silence_noise: list[float, float],  #! type list due to the format of parser.
    general_noise: list[float, float],
    prep_idmt: bool = False,
    prep_aam: bool = False,
    prep_maestro: bool = False,
) -> None:
    """
    Main entry point for preprocessing datasets.
    - Iterates through the provided dataset names and triggers the appropriate
    - processes function depending on the dataset type.
    - Skips processing if the output .npz file already exists and skipping is enabled.
    """
    for name in dataset_names:
        prep_msg = f'Skipping {name} – already preprocessed.'
        src = Path(__file__).resolve().parents[2] / 'data' / 'datasets' / name
        out = PREPROCESS_DIR / f'{name}.npz'
        if not src.exists():
            logging.error(f'{src} not found!')
            return

        logging.info(f'{name} preprocessing initiated:')
        if name == 'IDMT':
            if out.exists() and prep_idmt:
                logging.info(prep_msg)
                continue
            _preprocess_idmt_dataset(config.DATASETS[name], src, out, hop_len=idmt_hop_len, sample_rate=idmt_sr)
        elif name == 'AAM':
            if out.exists() and prep_aam:
                logging.info(prep_msg)
                continue
            try:  #! change noise args to tuple
                _preprocess_aam_dataset(src, out, tuple(silence_noise), tuple(general_noise))
            except KeyboardInterrupt:
                logging.warning('Interrupted. Cleaning up...')
                shutil.rmtree(PREPROCESS_DIR / '_temp_AAM', ignore_errors=True)
                raise
        elif name == 'MAESTRO':
            if out.exists() and prep_maestro:
                logging.info(prep_msg)
                continue
            _preprocess_maestro_dataset(config.DATASETS[name], src, out, tuple(silence_noise), tuple(general_noise))

    logging.info('Preprocessing finished.')


# ===========================
# === Filtering data prep ===
# ===========================


def _filter_idmt_dataset(
    data: dict,
    IDMT_size: int,
    guitar_ratio: float,
    output_path: Path,
    size_tracker: dict[str, int],
    use_max_size: bool = True,
) -> None:
    """
    Creates a filtered subset of the IDMT dataset or copy full dataset if use_max_size is True.
    """
    logging.info('IDMT filterling initiated:')
    if use_max_size:
        source_path = Path(PREPROCESS_DIR) / 'IDMT.npz'
        if source_path.exists():
            shutil.copy(source_path, output_path)
            logging.info(f'Max size - Copied full IDMT dataset to {output_path}')
        else:
            logging.error(f'Source not found: {source_path}')
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

    logging.info(f'Filtering IDMT dataset: {guitar_size:,} guitar samples | {nonguitar_size:,} non-guitar samples')
    rng = np.random.default_rng(config.SEED)
    guitar_indices = rng.choice(np.where(guitar_mask)[0], size=guitar_size, replace=False)
    nonguitar_indices = rng.choice(np.where(non_guitar_mask)[0], size=nonguitar_size, replace=False)
    selected_indices = np.concatenate([guitar_indices, nonguitar_indices])
    rng.shuffle(selected_indices)

    size_tracker['IDMT'] = len(X[selected_indices])
    save_npz(
        output_path,
        X=X[selected_indices],
        Y_chords=Y_chords[selected_indices],
        sources=sources[selected_indices],
        categories=categories[selected_indices],
    )


def _filter_aam_dataset(
    data: dict,
    AAM_size: int,
    AAM_instruments: list[str],
    output_path: Path,
    size_tracker: dict[str, int],
    use_max_size: bool = True,
    use_all_AAM_instruments: bool = False,
) -> None:
    """
    - Creates a filtered subset of AAM with specified instruments.
    - Slices entire subsets of AAM for each instrument if use_max_size is True
    """
    logging.info('AAM filterling initiated:')
    X = data['X']
    Y_chords = data['Y_chords']
    Y_notes = data['Y_notes']
    categories = data['categories'].astype(str)
    sources = data['sources'].astype(str)

    rng = np.random.default_rng(config.SEED)
    selected_indices = []

    if use_all_AAM_instruments:
        selected_indices = np.arange(len(X))
        if not use_max_size:
            if AAM_size > len(selected_indices):
                raise ValueError(f'Requested {AAM_size} samples, but only {len(selected_indices)} available.')
            selected_indices = rng.choice(selected_indices, size=AAM_size, replace=False)
        rng.shuffle(selected_indices)

    else:
        for inst in AAM_instruments:
            mask = categories == inst
            indices = np.where(mask)[0]

            if not use_max_size:
                available = len(indices)
                samples_per_instrument = AAM_size // len(AAM_instruments)
                if samples_per_instrument > available:
                    raise ValueError(
                        f'Requested {samples_per_instrument} samples for {inst}, but only {available} available.'
                    )
                indices = rng.choice(indices, size=samples_per_instrument, replace=False)
            selected_indices.extend(indices)
        rng.shuffle(selected_indices)

    if use_max_size:
        logging.info(f'Max size - Copied full AAM dataset to {output_path}')

    size_tracker['AAM'] = len(X[selected_indices])
    save_npz(
        output_path,
        X=X[selected_indices],
        Y_chords=Y_chords[selected_indices],
        Y_notes=Y_notes[selected_indices],
        sources=sources[selected_indices],
        categories=categories[selected_indices],
    )


def _filter_maestro_dataset(
    data: dict, MAESTRO_size: int, output_path: Path, size_tracker: dict[str, int], use_max_size: bool = True
) -> None:
    """
    Creates a filtered subset of MAESTRO or copy full dataset if use_max_size is True
    """
    logging.info('MAESTRO filterling initiated:')
    if use_max_size:
        source_path = PREPROCESS_DIR / 'MAESTRO.npz'
        shutil.copy(source_path, output_path)
        logging.info(f'Max size - Copied full MAESTRO dataset to {output_path}')
        return

    X = data['X']
    Y_notes = data['Y_notes']
    categories = data['categories'].astype(str)
    sources = data['sources'].astype(str)

    rng = np.random.default_rng(config.SEED)
    indices = np.arange(len(X))
    selected_indices = rng.choice(indices, size=MAESTRO_size, replace=False)
    rng.shuffle(selected_indices)

    size_tracker['MAESTRO'] = len(X[selected_indices])
    save_npz(
        output_path,
        X=X[selected_indices],
        Y_notes=Y_notes[selected_indices],
        sources=sources[selected_indices],
        categories=categories[selected_indices],
    )


def filter_data(
    dataset_names: list[str],
    filter_size: int,
    IDMT_ratio: float,
    AAM_ratio: float,
    MAESTRO_ratio: float,
    IDMT_guitar_ratio: float,
    AAM_instruments: list[str],
    use_max_size: bool = True,
    use_all_instruments: bool = False,
    filter_idmt: bool = False,
    filter_aam: bool = False,
    filter_maestro: bool = False,
):
    logging.info('Data filtering/slicing initiated:')
    (Path(__file__).resolve().parents[2] / 'data' / 'filtered').mkdir(parents=True, exist_ok=True)

    dataset_split_ratios = {'IDMT_ratio': IDMT_ratio, 'AAM_ratio': AAM_ratio, 'MAESTRO_ratio': MAESTRO_ratio}
    if not _validate_ratios(dataset_split_ratios, IDMT_guitar_ratio):
        return
    if not _validate_instruments(AAM_instruments):
        return

    IDMT_size, AAM_size, MAESTRO_size = [int(np.floor(filter_size * ratio)) for ratio in dataset_split_ratios.values()]
    sizes = {'IDMT': IDMT_size, 'AAM': AAM_size, 'MAESTRO': MAESTRO_size}
    fliter_flags = {
        'IDMT': filter_idmt,
        'AAM': filter_aam,
        'MAESTRO': filter_maestro,
    }
    for dataset_name in dataset_names:
        output_path = Path(__file__).resolve().parents[2] / 'data' / 'filtered' / f'{dataset_name}.npz'
        if output_path.exists() and fliter_flags.get(dataset_name, False):
            logging.info(f'Skipping {dataset_name} — filtered version already exists.')
            continue

        data = _load_dataset_to_filter(dataset_name)
        if not data:
            return

        if dataset_name == 'IDMT':
            _filter_idmt_dataset(
                data, IDMT_size, IDMT_guitar_ratio, output_path, use_max_size=use_max_size, size_tracker=sizes
            )
        elif dataset_name == 'AAM':
            _filter_aam_dataset(
                data,
                AAM_size,
                AAM_instruments,
                output_path,
                use_max_size=use_max_size,
                use_all_AAM_instruments=use_all_instruments,
                size_tracker=sizes,
            )
        elif dataset_name == 'MAESTRO':
            _filter_maestro_dataset(data, MAESTRO_size, output_path, use_max_size=use_max_size, size_tracker=sizes)
    logging.info('Data filtering/slicing complete.')
    logging.info(f'IDMT size: {sizes["IDMT"]:,} | AAM size: {sizes["AAM"]:,} | MAESTRO size: {sizes["MAESTRO"]:,}')


# ============================================
# === Stacking Datasets, Test-Train Splits ===
# ============================================


def stack_datasets(
    datasets: list[str] = ['IDMT', 'AAM', 'MAESTRO'],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Loads and stacks previously filtered datasets.
    """
    logging.info('Stackig and organizig filtered data...')
    datasets_dir = Path(__file__).resolve().parents[2] / 'data' / 'filtered'
    all_X, all_Y_chords, all_Y_notes, all_sources, all_datasets = [], [], [], [], []

    for dataset in datasets:
        ds_path = datasets_dir / f'{dataset}.npz'
        data = np.load(ds_path, allow_pickle=True)
        X = data['X']
        all_X.append(X)
        all_sources.extend(data['sources'])
        all_datasets.extend([dataset] * len(data['sources']))

        if 'Y_notes' in data:
            all_Y_notes.append(data['Y_notes'])
        else:
            all_Y_notes.append(np.full((len(X), SEQUENCE_LENGTH, 128), np.nan))

        if 'Y_chords' in data:
            all_Y_chords.append(data['Y_chords'])
        else:
            all_Y_chords.append(np.full((len(X), SEQUENCE_LENGTH), 'MISSING'))

    X = np.vstack(all_X)
    Y_notes = np.vstack(all_Y_notes)
    Y_chords = np.concatenate(all_Y_chords, axis=0)
    sources = np.array(all_sources)
    dataset_names = np.array(all_datasets)

    # Filter out samples with missing chord labels or all NaN notes
    chord_mask = Y_chords[:, 0] != 'MISSING'  # Only the first frame suffices
    note_mask = ~np.isnan(Y_notes[:, 0, 0])

    X_chords = X[chord_mask]
    Y_chords_filtered = Y_chords[chord_mask]
    sources_chords = sources[chord_mask]
    datasets_chords = dataset_names[chord_mask]
    X_notes = X[note_mask]
    Y_notes_filtered = Y_notes[note_mask]
    sources_notes = sources[note_mask]
    datasets_notes = dataset_names[note_mask]

    splitted_data = {
        'chords': {
            'X': X_chords,
            'Y': Y_chords_filtered,
            'sources': sources_chords,
            'datasets': datasets_chords,
        },
        'notes': {
            'X': X_notes,
            'Y': Y_notes_filtered,
            'sources': sources_notes,
            'datasets': datasets_notes,
        },
    }
    return splitted_data


def split_data(
    label_type: str, data: dict, train_ratio: float = 0.9, val_ratio: float = 0.05, test_ratio: float = 0.05
):
    """
    Splits the previously stacked datasets into test, train and validation sets.
    """
    logging.info(f'Splitting {label_type.upper()} data into test, train, val sets...')
    split_map_ratios = {
        'train': train_ratio,
        'val': val_ratio,
        'test': test_ratio,
    }
    if not _validate_ratios(split_map_ratios):
        return

    output_path = Path(__file__).resolve().parents[2] / 'data' / 'splitted'
    output_path.mkdir(parents=True, exist_ok=True)

    X = data[label_type]['X']
    Y = data[label_type]['Y']
    sources = data[label_type]['sources']
    datasets = data[label_type]['datasets']

    indices = np.arange(len(X))
    np.random.default_rng(config.SEED).shuffle(indices)

    train_end = int(train_ratio * len(indices))
    val_end = train_end + int(val_ratio * len(indices))
    test_end = val_end + int(test_ratio * len(indices))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:test_end]

    indices_lenghts = []
    for name in split_map_ratios.keys():
        output_file = output_path / f'{label_type}_{name}.npz'
        if name == 'train':
            selected_indices = train_indices
        elif name == 'val':
            selected_indices = val_indices
        elif name == 'test':
            selected_indices = test_indices

        save_npz(
            output_file,
            X=X[selected_indices],
            Y=Y[selected_indices],
            sources=sources[selected_indices],
            datasets=datasets[selected_indices]
        )
        indices_lenghts.append(len(selected_indices))
    logging.info(
        f'{label_type.upper()} -- train_size: {indices_lenghts[0]:,} | val_size: {indices_lenghts[1]:,} | test_size: {indices_lenghts[2]:,} -- Total: {len(X):,}'
    )


# =====================
# === Main function ===
# =====================


def main():
    setup_logger()
    parser = get_preprocess_parser()
    args = parser.parse_args()

    # Resolve force flags from list
    force_prep_idmt = 'IDMT' not in args.force_preprocess
    force_prep_aam = 'AAM' not in args.force_preprocess
    force_prep_maestro = 'MAESTRO' not in args.force_preprocess
    preprocess_data(
        dataset_names=args.datasets,
        idmt_hop_len=args.hop_length,
        idmt_sr=args.sample_rate,
        silence_noise=args.silence_noise,
        general_noise=args.general_noise,
        prep_idmt=force_prep_idmt,
        prep_aam=force_prep_aam,
        prep_maestro=force_prep_maestro,
    )
    force_filter_idmt = 'IDMT' not in args.force_filter
    force_filter_aam = 'AAM' not in args.force_filter
    force_filter_maestro = 'MAESTRO' not in args.force_filter
    filter_data(
        dataset_names=args.datasets,
        filter_size=args.filter_size,
        use_max_size=args.use_max_size,
        use_all_instruments=args.use_all_aam_instruments,
        IDMT_ratio=args.idmt_ratio,
        AAM_ratio=args.aam_ratio,
        MAESTRO_ratio=args.maestro_ratio,
        IDMT_guitar_ratio=args.idmt_guitar_ratio,
        AAM_instruments=args.aam_instruments,
        filter_idmt=force_filter_idmt,
        filter_aam=force_filter_aam,
        filter_maestro=force_filter_maestro,
    )
    data = stack_datasets(datasets=args.datasets)
    for type in ['notes', 'chords']:
        split_data(
            label_type=type,
            data=data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )


if __name__ == '__main__':
    main()
