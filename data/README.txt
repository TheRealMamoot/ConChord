NPZ FILE STRUCTURE
=====================================
------------------------------------------------------------------
IDMT DATASET
------------------------------------------------------------------
Each .npz file contains the following keys:

- X           : A 3D array of shape (num_sequences, sequence_len, 12)
                Contains chroma features extracted from audio.
                Each row is a normalized chroma vector for one frame.

- Y_chords     : A 2D array of shape (num_sequences, sequence_len)
                Contains chord labels aligned to each frame.

- sources     : A 1D array of shape (num_sequences,)
                Each entry is the string 'wav' to indicate source type.

- categories  : A 1D array of shape (num_sequences,)
                Each entry is either 'guitar' or 'non_guitar' to indicate the instrument category.

------------------------------------------------------------------
AAM DATASET
------------------------------------------------------------------
Each .npz file contains the following keys:

- X           : A 3D array of shape (num_sequences, sequence_len, 12)
                Contains chroma features extracted from MIDI using velocity-based encoding.

- Y_chords    : A 2D array of shape (num_sequences, sequence_len)
                Contains chord labels aligned to each frame.

- Y_notes     : A 3D binary array of shape (num_sequences, sequence_len, 128)
                Each row contains active MIDI note pitches (0–127) for the frame.

- sources     : A 1D array of shape (num_frames,)
                Each entry is the string 'midi' to indicate source type.

- categories  : A 1D array of shape (num_frames,)
                Each entry is the name of the instrument (e.g., 'AcousticGuitar', 'Ukulele', etc.)

------------------------------------------------------------------
MAESTRO DATASET
------------------------------------------------------------------
Each .npz file contains the following keys:

- X           : A 3D array of shape (num_sequences, sequence_len, 12)
                Contains chroma features extracted from MIDI using velocity-based encoding.

- Y_notes     : A 3D binary array of shape (num_sequences, sequence_len, 128)
                Each row contains active MIDI note pitches (0–127) for the frame.

- sources     : A 1D array of shape (num_frames,)
                Each entry is the string 'midi' to indicate source type.

- categories  : A 1D array of shape (num_frames,)
                Each entry is the name of the instrument which is piano.
                
------------------------------------------------------------------
NOTES
------------------------------------------------------------------
- All time alignment is based on fixed-length frames (e.g., hop_length/sample_rate).
- Sequence length is a sequence duration (default ~1s) devided by each frame length/duration 
- The chroma vectors are normalized along the pitch axis using L1 norm.
- The filtered directory contains the same data but filtered by user's inputs.
- The splitted directory contains test, train and validations sets with the source dataset for each sequence.