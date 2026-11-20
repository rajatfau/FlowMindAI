from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('data/sample/channel_profiles.csv')
fig, ax = plt.subplots(figsize=(6,4))
for case in [0,3,7]:
	sub = df[df.case_id == case]
	ax.plot(sub.u_m_s, sub.y_m*1000,
		label=f'Case {case}')
ax.set_xlabel('Velocity (m/s)')
ax.set_ylabel('Height (mm)')
ax.set_title('Analytical channel-flow profiles')
ax.legend()
fig.tight_layout()
path = Path('results/figures/channel_profiles.png')
fig.savefig(path, dpi=140)
print('Saved:', path)
