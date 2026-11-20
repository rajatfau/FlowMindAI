import pandas as pd
from pathlib import Path

def test_shape_and_missing():
    df = pd.read_csv('data/sample/channel_profiles.csv')
    assert df.shape == (520, 5)
    assert df.isna().sum().sum() == 0

def test_case_separation():
   
    root = Path('data/sample')
    ids = [set(pd.read_csv(root/f'{n}.csv').case_id)
    for n in ('train','val','test')]
    assert not (ids[0] & ids[1])
    assert not (ids[0] & ids[2])
    assert not (ids[1] & ids[2])

def test_no_slip_walls():
    df = pd.read_csv('data/sample/channel_profiles.csv')
    for _, group in df.groupby('case_id'):
        assert abs(group.iloc[0].u_m_s) < 1e-12
        assert abs(group.iloc[-1].u_m_s) < 1e-12
