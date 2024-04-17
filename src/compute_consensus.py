from misc import contour_maps, get_gnid_consensus, get_closest_gnid
from ib_color_naming.src.ib_naming_model import load_model
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import os 
PRECISION = 1e-20
ib_model = load_model()
need = ib_model.pM.flatten()
def mode_filter(data):
    y = data.argmax(axis=1)
    modemap = np.zeros_like(data)
    modemap[np.arange(y.size), y] = 1
    major = np.sum(modemap, axis=0) >= 10
    data = data[:, major] + 1e-20
    data = data / np.sum(data, axis=1).reshape(330,1)

    
    return data
def marginal(pXY, axis=1):
    return pXY.sum(axis)


def conditional(pXY):
    pX = pXY.sum(axis=1, keepdims=True)
    return np.where(pX > PRECISION, pXY / pX, 1 / pXY.shape[1])

def xlogx(v):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(v > PRECISION, v * np.log2(v), 0)


def entropy(p, axis=None):
    return -xlogx(p).sum(axis=axis)


def mutual_information(pXY):
    return entropy(pXY.sum(axis=0)) + entropy(pXY.sum(axis=1)) - entropy(pXY)

def kl(p, q):
    return (xlogx(p) - np.where(p > PRECISION, p * np.log2(q + PRECISION), 0)).sum()

path = 'figures/consensus_maps/'

os.makedirs(path + '/ib', exist_ok=True)
os.makedirs(path + '/wcs', exist_ok=True)
os.makedirs(path + '/nil', exist_ok=True)
os.makedirs(path + '/il', exist_ok=True)
os.makedirs(path + '/rl', exist_ok=True)
os.makedirs(path + '/rm', exist_ok=True)
os.makedirs(path + '/wcs_maps/', exist_ok=True)

with open('data/wcs_lang.pkl', 'rb') as f:
    wcs = pickle.load(f)
plt.rcParams["figure.figsize"] = (15, 5)
for l in wcs:
    mapping = mode_filter(wcs[l])
    contour_maps(mapping, ib_model)
    plt.savefig(path + f'/wcs_maps/{l}_{mapping.shape[1]}.pdf')
    plt.close()


ib_systems = ib_model.qW_M
ib_systems = [mode_filter(e) for e in ib_systems]
plt.rcParams["figure.figsize"] = (15, 5)
for i in range(1, len(ib_systems)):
    if ib_systems[i].shape[1] != ib_systems[i-1].shape[1]:
        contour_maps(ib_systems[i-1], ib_model)
        print(f'number of words {ib_systems[i-1].shape[1]}')
        plt.savefig(path + f'/ib/ib_{ib_systems[i-1].shape[1]}_term.pdf')
        plt.close()
    if ib_systems[i].shape[1] > 7:
        break


with open('iterated_learning_exp_cogsci/NIL/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
nil_consensus = get_gnid_consensus(encoders, need)
for i in range(3,  7):
    contour_maps(nil_consensus[i], ib_model)
    plt.savefig(path + f'/nil/nil_{i}_term.pdf')
    plt.close()


with open('data/wcs_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
wcs_consensus = get_gnid_consensus(encoders, need)
for i in range(3,  7):
    contour_maps(wcs_consensus[i], ib_model)
    plt.savefig(path + f'/wcs/wcs_{i}_term.pdf')
    plt.close()

with open('iterated_learning_exp_cogsci/IL/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
il_consensus = get_gnid_consensus(encoders, need)
for i in range(3,  7):
    contour_maps(il_consensus[i], ib_model)
    plt.savefig(path + f'/il/il_{i}_term.pdf')
    plt.close()

with open('iterated_learning_exp_cogsci/RL/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
rl_consensus = get_gnid_consensus(encoders, need)
for i in range(3,  7):
    contour_maps(rl_consensus[i], ib_model)
    plt.savefig(path + f'/rl/rl_{i}_term.pdf')
    plt.close()

with open('iterated_learning_exp_cogsci/random_model/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
rm_consensus = get_gnid_consensus(encoders, need)
for i in range(3,  7):
    contour_maps(rm_consensus[i], ib_model)
    plt.savefig(path + f'/rm/rm_{i}_term.pdf')
    plt.close()


# explore 5 terms closer
with open('iterated_learning_exp_cogsci/NIL/encoders.pkl', 'rb') as f:
    nil_encoders = pickle.load(f)
from misc import gNID
v5encoders = []
v5comp = []
for e in nil_encoders:
    argmax = np.argmax(e, axis=1)
    unique = len(np.unique(argmax))
    if unique == 5:
        v5encoders.append(e)
spkr_sim = np.zeros((len(v5encoders), len(v5encoders)))
for i, e1 in enumerate(v5encoders):
    for j, e2 in enumerate(v5encoders):
        spkr_sim[i, j] = 1 - gNID(e1, e2, ib_model.pM)

from sklearn.cluster import SpectralClustering
n_clusters = 3
sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=100, assign_labels='kmeans', random_state=0)
labels = sc.fit_predict(spkr_sim)
clustering = {0:[], 1:[], 2:[]}
for i, e in enumerate(v5encoders):
    clustering[labels[i]].append(e)

gnid_consensus1 = get_gnid_consensus(clustering[0], need)
contour_maps(gnid_consensus1[5], ib_model)
plt.savefig(path + f'/nil/nil_5_term_0.pdf')
plt.close()
gnid_consensus2 = get_gnid_consensus(clustering[1], need)
contour_maps(gnid_consensus2[5], ib_model)
plt.savefig(path + f'/nil/nil_5_term_1.pdf')
plt.close()
gnid_consensus3 = get_gnid_consensus(clustering[2], need)
contour_maps(gnid_consensus3[5], ib_model)
plt.savefig(path + f'/nil/nil_5_term_2.pdf')
plt.close()

