from pathlib import Path
import numpy as np
import pandas as pd

def test_predictions():
    df = pd.read_csv('data/sample/first_predictions.csv')
    assert len(df) == 65
    assert np.isfinite(df.prediction_m_s).all()

def test_checkpoint():
    assert Path('models/first_channel_mlp.pt').is_file()