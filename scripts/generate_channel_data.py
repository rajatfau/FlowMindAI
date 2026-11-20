from pathlib import Path
import numpy as np
import pandas as pd
out = Path('data/sample')
out.mkdir(parents=True, exist_ok=True)
H = 0.001
nu = 1e-6
y = np.linspace(0, H, 65)
rows = []
for case, mean_u in enumerate(np.arange(1, 9) / 100):
    u = 6 * mean_u * (y/H) * (1-y/H)
    for yy, uu in zip(y, u):
        rows.append((case, mean_u, mean_u*H/nu,
yy, uu))
cols = ['case_id', 'u_mean_m_s', 'reynolds_H',
'y_m', 'u_m_s']
df = pd.DataFrame(rows, columns=cols)
df.to_csv(out/'channel_profiles.csv', index=False)
print('Cases:', df.case_id.nunique())
print('Rows:', len(df))
print('Saved: data/sample/channel_profiles.csv')
