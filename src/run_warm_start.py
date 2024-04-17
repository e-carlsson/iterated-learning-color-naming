import os
import glob
import pickle

dir = 'ib_warm_start/files/'
exp = glob.glob(dir + '*.pkl')

for e in exp:
    os.system(f"sbatch python_script.sh warm_start_exp.py {e}")