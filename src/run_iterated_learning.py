import os
import pickle
import time
from misc import get_random_model
from ib_color_naming.src.ib_naming_model import load_model

'''
Example of how to initialize on a slurm cluster
'''
hidden_dim = 100
layers = 2
gamma = 0.001
dir = f'iterated_learning_exp_hidden_dim_{hidden_dim}_layers_{layers}/'
os.makedirs(dir, exist_ok=True)
nil = dir + 'NIL/files/'
os.makedirs(nil, exist_ok=True)
il = dir + 'IL/files/'
os.makedirs(il, exist_ok=True)
rl = dir + 'RL/files/'
os.makedirs(rl, exist_ok=True)

with open('data/wcs_encoders.pkl', 'rb') as f:
    wcs = pickle.load(f)

ib_model = load_model()
get_random_model(wcs, ib_model=ib_model, path=dir)


vocab_sizes = list(range(3, 11)) + [100]
n_runs = range(100)

# Submit NIL jobs
for v in vocab_sizes:
    for i in n_runs:
        save_path = nil + f'vocab_size_{v}_run_{i}.pkl'
        os.system(f"sbatch python_script.sh iterated_learning.py {save_path} {v} NIL {hidden_dim} {layers} {gamma}")

time.sleep(30) # Avoid submitting too many jobs at once

# Submit IL jobs
for v in vocab_sizes:
    for i in n_runs:
        save_path = il + f'vocab_size_{v}_run_{i}.pkl'
        os.system(f"sbatch python_script.sh iterated_learning.py {save_path} {v} IL {hidden_dim} {layers} {gamma}")


time.sleep(30) # Avoid submitting too many jobs at once

# Submit RL jobs
for v in vocab_sizes:
    for i in n_runs:
        save_path = rl + f'vocab_size_{v}_run_{i}.pkl'
        os.system(f"sbatch python_script.sh iterated_learning.py {save_path} {v} RL {hidden_dim} {layers} {gamma}")
