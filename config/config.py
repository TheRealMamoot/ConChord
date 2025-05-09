AUDIO_PARAMS = {
    "sample_rate": 11025,
    "hop_length": 512
}

DATASETS = {
    "IDMT": {
      "url": ["https://zenodo.org/records/7544213/files/IDMT-SMT-CHORDS.zip?download=1"],
      "subdirs": ["guitar", "non_guitar"]
    },
    "AAM": {
      "url": ["https://zenodo.org/records/5794629/files/0001-1000-annotations-v1.1.0.zip?download=1",
              "https://zenodo.org/records/5794629/files/0001-1000-midis.zip?download=1"]
      }
}
