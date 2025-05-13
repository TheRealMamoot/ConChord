import os
from pathlib import Path
import numpy as np

def load_lab_file(path: Path) -> list[tuple[float, float, str]]:
    segments = []
    with path.open('r') as f:
        for line in f:
            parts = line.strip().split()
            start, end, chord = parts
            segments.append((float(start), float(end), chord))
    return segments

def align_labels_to_frames(frame_times: np.ndarray, lab_segments: list[tuple[float, float, str]]) -> list[str]:
    labels = []
    for t in frame_times:
        matched_label = 'N'  # Default to 'N' (no chord)
        for start, end, chord in lab_segments:
            if start <= t < end:
                matched_label = chord
                break
        labels.append(matched_label)
    return labels

def convert_arff_to_lab(arff_path: Path, output_path: Path):
    with arff_path.open('r') as f:
        lines = f.readlines()

    chord_entries = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 4:
            continue
        try:
            time = float(parts[0])
            chord = parts[3].strip().strip("'")
            chord_entries.append((time, chord))
        except ValueError:
            continue

    # Convert to intervals
    lab_intervals = []
    for i in range(len(chord_entries) - 1):
        start_time = chord_entries[i][0]
        end_time = chord_entries[i + 1][0]
        chord = chord_entries[i][1]
        lab_intervals.append((start_time, end_time, chord))

    # Write .lab format
    with output_path.open('w') as out:
        for start, end, chord in lab_intervals:
            out.write(f'{start:.6f} {end:.6f} {chord}\n')

def folder_is_populated(path: Path) -> bool:
    return any(p.name != '.gitkeep' for p in path.iterdir())