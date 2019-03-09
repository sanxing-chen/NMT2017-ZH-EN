#%%
import matplotlib
from matplotlib import pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import json
from pathlib import Path

#%%
lines = Path('./train.log').read_text().splitlines()

losses = []
nll_losses = []
ppls = []
lrs = []

for line in lines:
    if line.startswith('{'):
        data = json.loads(line, parse_float=float)
        losses.append(float(data['loss']))
        nll_losses.append(float(data['nll_loss']))
        ppls.append(float(data['ppl']))
        lrs.append(float(data['lr']))

#%%
fig, axs = plt.subplots(2, 2)
axs = axs.flatten()
axs[0].plot(losses)
axs[0].set_title('loss')
axs[1].plot(nll_losses)
axs[1].set_title('nll_loss')
axs[2].plot(ppls)
axs[2].set_title('ppl')
axs[3].plot(lrs)
axs[3].set_title('lr')
plt.tight_layout()

"""
data format
{
    "epoch": 1,
    "update": 0.04648568240981778,
    "loss": "10.902",
    "nll_loss": "10.227",
    "ppl": "1198.23",
    "wps": 14907,
    "ups": "0.6",
    "wpb": 22246,
    "bsz": 825,
    "num_updates": 1001,
    "lr": 0.000774775,
    "gnorm": "0.476",
    "clip": "0%",
    "oom": 0.0,
    "wall": 1682,
    "train_wall": 1467
}
"""