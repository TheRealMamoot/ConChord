NPZ FILE STRUCTURE
=====================================
This file describes the structure of the .npz files generated for training and evaluation.
------------------------------------------------------------------
IDMT DATASET
------------------------------------------------------------------
Each .npz file contains the following keys:

- X           : A 2D array of shape (num_frames, 12)
                Contains chroma features extracted from audio.
                Each row is a normalized chroma vector for one frame.

- Y_chords     : A 1D array of shape (num_frames,)
                Contains chord labels aligned to each frame.

- sources     : A 1D array of shape (num_frames,)
                Each entry is the string 'wav' to indicate source type.

- categories  : A 1D array of shape (num_frames,)
                Each entry is either 'guitar' or 'non_guitar' to indicate the instrument category.

------------------------------------------------------------------
AAM DATASET
------------------------------------------------------------------
Each .npz file contains the following keys:

- X           : A 2D array of shape (num_frames, 12)
                Contains chroma features extracted from MIDI using velocity-based encoding.

- Y_chords    : A 1D array of shape (num_frames,)
                Contains chord labels aligned to each frame.

- Y_notes     : A 2D binary array of shape (num_frames, 128)
                Each row contains active MIDI note pitches (0–127) for the frame.

- sources     : A 1D array of shape (num_frames,)
                Each entry is the string 'midi' to indicate source type.

- categories  : A 1D array of shape (num_frames,)
                Each entry is the name of the instrument (e.g., 'AcousticGuitar', 'Ukulele', etc.)

------------------------------------------------------------------
NOTES
------------------------------------------------------------------
- All time alignment is based on fixed-length frames (e.g., hop_length/sample_rate).
- Files are compressed using numpy's savez_compressed.
- The chroma vectors are normalized along the pitch axis using L1 norm.