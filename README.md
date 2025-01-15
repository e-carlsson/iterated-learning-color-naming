# Neural Iterated Learning and Color Naming

Code used for the experiments in [Cultural evolution via iterated learning and communication explains efficient color naming systems](https://academic.oup.com/jole/article/9/1-2/49/7907230).

If you find this code useful, please consider citing: 

@article{carlsson2024cultural,
  title={Cultural evolution via iterated learning and communication explains efficient color naming systems},
  author={Carlsson, Emil and Dubhashi, Devdatt and Regier, Terry},
  journal={Journal of Language Evolution},
  volume={9},
  number={1-2},
  pages={49--66},
  year={2024},
  publisher={Oxford University Press UK}
}

Code related to IB and color naming comes from https://github.com/nogazs/ib-color-naming.

Neural Iterated Learning algorithm was introduced in https://arxiv.org/pdf/2002.01365.pdf


To run a NIL experiment:
1. cd src/
2. pip install -r requirements.txt
3. git submodule add https://github.com/nogazs/ib-color-naming.git ib_color_naming
4. python iterated_learning.py il_experiment.pkl 100 NIL

To run the main experiment on a slurm custer (it takes about 30 min on a >1000cpu system):
1. Specify details in python_script.sh
2. run: sbatch python_script.sh run_iterated_learning.py


The NIL can be initialized from a particular system 
