This repo contains code for our paper [Learning Belief Representations for Imitation Learning in POMDPs](https://arxiv.org/abs/1906.09510) published at UAI 2019. 

The code was tested with the following packages:

* python 3.6.6
* pytorch 0.4.1
* gym  0.10.8

## Running command
To run MuJoCo experiments, use the script **run_mujoco.sh** with the following usage:

```
bash run_mujoco.sh [env] [belief_loss_type] [belief_regularization]
```

BMIL results can be reproduced with *bash run_mujoco.sh [env] task_aware True*

### Expert trajectories

Please update the path to expert trajectories in the file "code/conf/envParams.yaml". Also see the storage requirements in "code/expert_envs.py" and modify as per convenience. 


## Credits
The base for this code is provided by [DVRL](https://github.com/maximilianigl/DVRL), which itself utilizes methods from [this](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail). We also use [OpenAI baselines](https://github.com/openai/baselines) helpers for vectorized environments.
