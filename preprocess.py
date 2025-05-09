import logging
import os

from librosa.feature import chroma_stft
from librosa import load, frames_to_time
from mido.midifiles.meta import KeySignatureError
import numpy as np
from pretty_midi import PrettyMIDI
import shutil
from tqdm import tqdm

from config.config import DATASETS, AUDIO_PARAMS
from utils.logger import setup_logging
from utils.parser import get_preprocess_parser
from utils.utils import load_lab_file, align_labels_to_frames, convert_arff_to_lab

BASE_DIR  = 'data/processed/'
FRAME_DURATION = AUDIO_PARAMS['hop_length'] / AUDIO_PARAMS['sample_rate']

def preprocess(dataset_names: list[str]):

    for dataset_name in dataset_names:
        dataset = DATASETS[dataset_name]
        src_dir = f'data/datasets/{dataset_name}'
        if not os.path.exists(src_dir):
            logging.error(f'{src_dir} directory does not exist!')
            return
        
        processed_dir = os.path.join(BASE_DIR, dataset_name)
        os.makedirs(processed_dir, exist_ok=True)
        logging.info(f'{dataset_name} preprocess initiated.')

        if dataset_name=='IDMT': # Contains .wav and .lab files
            sub_dirs = dataset.get('subdirs')
            for dir in sub_dirs:
                sub_dir = os.path.join(src_dir, dir)
                wav_files = [f for f in os.listdir(sub_dir) if f.endswith('.wav')]
                annotations = os.path.join(sub_dir, f'{dir}_annotation.lab')
                lab_segments = load_lab_file(annotations)

                for wav_file in tqdm(wav_files,
                                        desc=f'{dir}',
                                        unit='file',
                                        ncols=80,
                                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} @ {rate_fmt}]'):
                    
                    npz_name = wav_file.replace('.wav', '.npz')
                    output_path = os.path.join(processed_dir, dir, npz_name)
                    if os.path.exists(output_path):
                        logging.info(f'Skipping {npz_name} – already processed at {dir}')
                        continue

                    wav_file_path = os.path.join(sub_dir, wav_file)
                    audio, sr = load(wav_file_path, sr=AUDIO_PARAMS.get('sample_rate'))
                    chroma = chroma_stft(y=audio, sr=sr, hop_length=AUDIO_PARAMS.get('hop_length')).T
                    frame_times = frames_to_time(range(chroma.shape[0]), sr=sr, hop_length=AUDIO_PARAMS.get('hop_length'))
                    chord_labels = align_labels_to_frames(frame_times, lab_segments)

                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    np.savez_compressed(
                        output_path,
                        chroma=chroma,
                        frame_times=frame_times,
                        chord_labels=chord_labels
                    )

        if dataset_name=='AAM': # Contains .mid and .arff files
            preprocess_temp = os.path.join(processed_dir, f'_temp_{dataset_name}')
            os.makedirs(preprocess_temp, exist_ok=True)
            mid_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.mid') and 'Drums' not in f])
            arff_files = sorted([f for f in os.listdir(src_dir) if f.endswith('beatinfo.arff')])

            for arff_file in arff_files:
                arff_file_path = os.path.join(src_dir, arff_file)
                output_lab_path = os.path.join(preprocess_temp, arff_file.replace('.arff', '.lab'))
                if os.path.exists(output_lab_path):
                    logging.info(f'Skipping {arff_file} – already processed.')
                    continue
                convert_arff_to_lab(arff_file_path, output_lab_path)
            logging.info(f'.lab files created.')

            for midi in tqdm(mid_files,
                             desc=f'{dataset_name}',
                             unit='file',
                             ncols=80,
                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} @ {rate_fmt}]'):
                midi_id = midi[:4]
                midi_path = os.path.join(src_dir, midi)
                npz_name = midi.replace('.mid', '.npz')
                output_path = os.path.join(processed_dir, npz_name)

                if os.path.exists(output_path):
                    logging.info(f'Skipping {midi} – already processed.')
                    continue

                try:
                    midi_data = PrettyMIDI(midi_path)
                except KeySignatureError:
                    logging.warning(f'Skipped - Invalid key signature in {midi_path}')
                    continue  

                lab_file = os.path.join(preprocess_temp, f'{midi_id}_beatinfo.lab')
                lab_segments = load_lab_file(lab_file)

                if not os.path.exists(lab_file):
                    logging.warning(f'Missing .lab file for {midi_id}, skipping.')
                    continue

                end_time = midi_data.get_end_time()
                frame_times = np.arange(0, end_time, FRAME_DURATION)

                # Manual note velocity extraction
                chroma = np.zeros((len(frame_times), 12))
                for note in midi_data.instruments[0].notes:
                    start_idx = np.searchsorted(frame_times, note.start)
                    end_idx = np.searchsorted(frame_times, note.end)
                    pitch_class = note.pitch % 12
                    velocity = note.velocity

                    chroma[start_idx:end_idx, pitch_class] += velocity

                chord_labels = align_labels_to_frames(frame_times, lab_segments)

                np.savez_compressed(
                    output_path,
                    chroma=chroma,
                    frame_times=frame_times,
                    chord_labels=chord_labels
                )
            shutil.rmtree(preprocess_temp)
            
def main():
    setup_logging()
    parser = get_preprocess_parser()
    args = parser.parse_args()
    preprocess(dataset_names=args.datasets)

if __name__ == '__main__':
    main()