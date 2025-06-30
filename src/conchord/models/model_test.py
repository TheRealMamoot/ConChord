from pathlib import Path

import numpy as np
import pandas as pd
from keras import Model
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

from src.conchord.models.models import load_data
from src.conchord.utils.parser import get_model_parser

if __name__ == '__main__':
    parser = get_model_parser()
    args = parser.parse_args()
    target_dir = Path(__file__).resolve().parents[0]
    data = load_data(model_type=args.model_type, allowed_datasets=args.datasets)
    encoder: LabelEncoder = data['encoder']
    model: Model = load_model(str(target_dir / f'{args.model_type}.keras'), compile=False)

    Y_pred_prob = model.predict(data['X_test'])
    Y_pred: np.ndarray = np.argmax(Y_pred_prob, axis=-1)
    Y_true: np.ndarray = np.argmax(data['Y_test'], axis=-1)

    num_samples, seq_len = data['Y_test'].shape[:2]
    Y_pred = encoder.inverse_transform(Y_pred.flatten()).reshape(num_samples, seq_len)
    Y_true = encoder.inverse_transform(Y_true.flatten()).reshape(num_samples, seq_len)

    n_seqs, n_frames = Y_true.shape
    sources = np.repeat(data['sources_test'], n_frames).reshape(n_seqs, n_frames)
    datasets = np.repeat(data['datasets_test'], n_frames).reshape(n_seqs, n_frames)

    rows = []
    for seq_idx in range(n_seqs):
        for frame_idx in range(n_frames):
            rows.append(
                {
                    'sequence_idx': seq_idx,
                    'frame_idx': frame_idx,
                    'true': Y_true[seq_idx, frame_idx],
                    'predicted': Y_pred[seq_idx, frame_idx],
                    'dataset': datasets[seq_idx, frame_idx],
                    'source': sources[seq_idx, frame_idx],
                }
            )

    df_predictions = pd.DataFrame(rows)
    df_predictions.to_csv(str(target_dir / f'{args.model_type}.csv'), index=False)

    result = (
        df_predictions.groupby(['sequence_idx'])
        .agg(
            true=('true', lambda x: x.mode().iloc[0]),
            predicted=('predicted', lambda x: x.mode().iloc[0]),
        )
        .assign(correct=lambda x: x['predicted'] == x['true'])
    ).reset_index()
    acc = result['correct'].sum() / len(result['correct'])
    print(f'Accuracy: {acc:.2f}')

    
