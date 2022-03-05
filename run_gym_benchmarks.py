import os 
import time

GYM_ENVS = [
  'ContinuousCartPoleEnv', 
  "MountainCarContinuous-v0",
  "Pendulum-v0",
  "BipedalWalker-v3",
  "BipedalWalkerHardcore-v3",
  "LunarLanderContinuous-v2",
  'Ant-v3', 
  'Walker2d-v3', 
  'HalfCheetah-v3', 
  'Hopper-v3'
]
SETTINGS = ['DPG', 'PPO', 'RACER', 'VRACER',]
TIMEOUT = 30 * 60
for env in GYM_ENVS:
  for sets in SETTINGS:
    cmd = f'smarties.py {env} {sets}.json --gym -r {env}-{sets} --timeout {TIMEOUT}'
    print(f'RUNNING {cmd}')
    os.system(cmd)
    time.sleep(TIMEOUT*0.05)