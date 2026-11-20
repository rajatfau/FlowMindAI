import pandas as pd
from pathlib import Path
root = Path('data/sample')
df = pd.read_csv(root/'channel_profiles.csv')
splits = {'train': [0,1,2,3,4,5],
'val': [6], 'test': [7]}
for name, cases in splits.items():
	part = df[df.case_id.isin(cases)]
	part.to_csv(root/f'{name}.csv', index=False)
	print(name, len(part), 'rows')
