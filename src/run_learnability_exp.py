import os
import glob
import pickle

dir = 'learnability_exp/models/'
exp = glob.glob(dir + '*.pkl')

for e in exp:
    with open(e, 'rb') as f:
        d = pickle.load(f)
        os.system(f"sbatch python_script.sh learnability_exp.py {e}")