from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass(frozen=True)
class Config:
    SEED: int = 42

    AUDIO_PARAMS: Dict[str, int] = field(default_factory=lambda: {
        "sample_rate": 11025,
        "hop_length": 512
    })

    DATASETS: Dict[str, Dict] = field(default_factory=lambda: {
        "IDMT": {
            "url": ["https://zenodo.org/records/7544213/files/IDMT-SMT-CHORDS.zip?download=1"],
            "subdirs": ["guitar", "non_guitar"]
        },
        "AAM": {
            "url": [
                "https://zenodo.org/records/5794629/files/0001-1000-annotations-v1.1.0.zip?download=1",
                "https://zenodo.org/records/5794629/files/0001-1000-midis.zip?download=1"
            ]
        },
        "MAESTRO": {
            "url": ["https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"],
            "subdirs": ["2018", "2017"]
        }
    })

    INSTRUMENTS: Dict[str, List[str]] = field(default_factory=lambda: {
        "AAM": [
            'AcousticGuitar','AltoSax','Balalaika','BrightPiano','Cello','Clarinet',
            'DoubleBassArco','DoubleBassPizz','ElectricBass','ElectricGuitarClean','ElectricGuitarCrunch',
            'ElectricGuitarLead','ElectricPiano','Erhu','Flugelhorn','Flute','Fujara','Jinghu','MorinKhuur',
            'OrganBass','PanFlute','Piano','Shakuhachi','Sitar','TenorSax','Trombone','Trumpet',
            'Ukulele','Viola','Violin'
        ]
    })

    DEFAULTS: Dict[str, Any] = field(default_factory=lambda: {
        "downloader": {
            "datasets": ['IDMT','AAM','MAESTRO'],
            "chunk-size": 256
        },
        "preprocess": {
            "datasets": ['IDMT','AAM','MAESTRO'],
            "filter-size": 300_000,
            "idmt-ratio": 0.3334,
            "aam-ratio": 0.3333,
            "maestro-ratio": 0.3333,
            "idmt-guitar-ratio": 0.5,
            "aam-instruments": ['AcousticGuitar', 'Piano']
        }
    })