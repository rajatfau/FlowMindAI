from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('data/sample/first_predictions.csv')
fig, ax = plt.subplots(figsize=(6,4))
y = df.y_m.to_numpy()*1000
ax.plot(df.u_m_s,y,label='Analytical reference')
ax.plot(df.prediction_m_s,y,'--',label='Neural network')
ax.set_xlabel('Velocity (m/s)')
ax.set_ylabel('Channel height (mm)')
ax.set_title('Held-out flow case')
ax.legend()
fig.tight_layout()
path = Path('results/figures/first_prediction.png')
path.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(path,dpi=140)
print('Saved:',path)