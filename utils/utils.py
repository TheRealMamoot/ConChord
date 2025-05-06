import os

def load_lab_file(path: str) -> list:
    chord_labels = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, chord = parts
                chord_labels.append((float(start), float(end), chord))
    return chord_labels

def folder_is_populated(path: str) -> bool:
    contents = os.listdir(path)
    return any(f for f in contents if f != '.gitkeep')