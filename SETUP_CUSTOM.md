# Setup

## Isaac Lab Enviornment
It is recommended to setup a conda env. For this, I created ```env_isaaclab``` as descirbed in the official Issac Lab docs:
```bash
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
```

## For Sim2Sim Transfer
I did not follow the installation guide in ```doc/setup_en.md``` for just testing in Mujoco. For staters, install ```mujoco```, ```yaml```, and ```pygame``` (some of this may already be included if you setup ```env_isaaclab```).

Then, I just ran:
```bash
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```
from the repo root directory. 