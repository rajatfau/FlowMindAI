from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

torch.manual_seed(42)
root = Path('data/sample')
train = pd.read_csv(root/'train.csv')
val = pd.read_csv(root/'val.csv')
test = pd.read_csv(root/'test.csv')

def convert(df):
    x = np.column_stack([df.u_mean_m_s / 0.08, df.y_m / 0.001])
    y = df.u_m_s.to_numpy()[:, None] / 0.12
    return (torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32))

xtr, ytr = convert(train)
xva, yva = convert(val)
xte, yte = convert(test)

model = nn.Sequential(nn.Linear(2,32), nn.Tanh(), nn.Linear(32,32), nn.Tanh(),nn.Linear(32,1))
optimizer = torch.optim.Adam(model.parameters(),lr=0.005)
loss_fn = nn.MSELoss()
checkpoint = Path('models/first_channel_mlp.pt')
checkpoint.parent.mkdir(exist_ok=True)
best = float('inf')
for epoch in range(1,1001):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(xtr),ytr)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        vloss = loss_fn(model(xva),yva).item()
    if vloss < best:
        best = vloss
    torch.save(model.state_dict(),checkpoint)
    if epoch in (1,250,500,750,1000):
        print(f'Epoch {epoch}: val MSE={vloss:.6f}')

model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
model.eval()
with torch.no_grad():
    predicted = model(xte).numpy().ravel()*0.12
actual = test.u_m_s.to_numpy()
mae = np.mean(np.abs(predicted-actual))
rmse = np.sqrt(np.mean((predicted-actual)**2))
print(f'Test MAE (m/s): {mae:.6f}')
print(f'Test RMSE (m/s): {rmse:.6f}')
print('Saved model:',checkpoint)
result = test.copy()
result['prediction_m_s'] = predicted
result.to_csv(root/'first_predictions.csv',index=False)