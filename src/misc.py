import matplotlib.pyplot as plt
import numpy as np
from ib_color_naming.src.tools import gNID, H, MI

PRECISION = 1e-32
from scipy.spatial import distance_matrix
import pandas as pd
from skimage import color
from scipy.special import softmax
import pickle

munsell = pd.read_csv("data/munsell_chart.txt", sep="\t")
cielab = pd.read_csv("data/munsell_chart.txt", sep="\t")[["L*", "a*", "b*"]].values
import sys
import os
from time import time


def get_expected_reward(speaker, need, reward_matrix):
    """
    Compute the expected reward of speaker model assuming optimal listener
    """
    listener = bayes_listener(speaker, need)
    optimal_rewards = []
    for w in range(speaker.shape[1]):
        qMW = []
        for i in range(330):
            rMW = (listener[:, w] * reward_matrix[:, i]).sum()
            qMW.append(rMW)
        optimal_rewards.append(np.max(qMW))

    pW = (speaker * need.reshape(330, 1)).sum(axis=0)
    return (pW * np.array(optimal_rewards)).sum()


# Stuff for convexity
from scipy.spatial import Delaunay, ConvexHull
from scipy.spatial.distance import cdist, pdist
from scipy.optimize import linprog
from copy import deepcopy


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is `MxK` array of the
    coordinates of `M` points in `K`dimensions
    """

    # Check if hull is 2 or 3 dimensional
    x0 = hull[0, 0]

    if hull.shape[0] < 4 or np.all(hull[:, 0] == x0):
        # Scipy needs at least 4 points to construct a hull, run slower approach
        return [point_in_hull_slow(hull, point) for point in p]

    hull = ConvexHull(hull)
    return [point_in_hull(hull, point) for point in p]


def point_in_hull_slow(points, x):
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success


def point_in_hull(hull, point, tolerance=1e-12):
    return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)


def get_convexity(speaker):
    """
    Check convexity of individual using the measure of Shane and Jakub
    :param modemap:
    :return:
    """
    y = speaker.argmax(axis=1)
    modemap = np.zeros_like(speaker)
    modemap[np.arange(y.size), y] = 1
    w_conv = []
    w_sizes = []
    unique = np.unique(y)
    for w in unique:
        w_args = modemap[:, w] == 1
        if np.sum(w_args) > 1:
            points = cielab[w_args]
            contained = in_hull(cielab, points)
            total = np.sum(contained)
            w_points = np.sum(w_args)
            if (
                w_points > total
            ):  # The hull-checker sometimes fails to include points on the boundary. w_points can never we larger than total
                w_points = total
                print("Failed to include points on boundary")
                raise ValueError
            conv = (w_points**2) / total
        else:
            conv = 1
            w_points = 1
        w_sizes.append(w_points)
        w_conv.append(conv)
    w_sizes = np.array(w_sizes)
    if np.sum(w_sizes) != 330:
        print("Error in convexity")
        raise ValueError
    conv = np.sum(w_conv) / np.sum(w_sizes)  # np.sum(w_sizes * np.array(w_conv))
    return np.round(conv, decimals=3)


def get_soft_convexity(dist, need):
    _, w_range = dist.shape
    fraction = []
    y = dist.argmax(axis=1)
    modemap = np.zeros_like(dist)
    modemap[np.arange(y.size), y] = 1
    for w in range(w_range):
        w_args = modemap[:, w] == 1
        if np.sum(w_args) > 1:
            points = cielab[w_args]
            contained = in_hull(cielab, points)
            total_mass = np.sum(need[contained])
            correct_mass = np.sum(need[contained].flatten() * dist[contained, w])
            fraction.append(correct_mass / total_mass)
        elif np.sum(w_args) == 1:
            total_mass = np.sum(need[w_args])
            correct_mass = np.sum(need[w_args].flatten() * dist[w_args, w])
            fraction.append(correct_mass / total_mass)
        else:
            fraction.append(0)  # p(w) will also be zero

    pW = np.sum(dist * need.reshape(330, 1), axis=0)
    return np.around(np.sum(pW * np.array(fraction)), decimals=3)


def get_conditional_entropy(speaker, need):
    """
    Compute the conditional entropy H(W|M) of the speaker model
    """
    entr = 0
    for i in range(330):
        entr += need[i] * (speaker[i, :] * np.log2(speaker[i, :] + PRECISION)).sum()

    return -entr


def get_entropy(speaker, need):
    """
    Compute the entropy H(W) of the speaker model
    """
    pW = (speaker * need.reshape(330, 1)).sum(axis=0)
    return -(pW * np.log2(pW + PRECISION)).sum()


# NID and SoftNID as defined in Zaslavsky et al 2018
# Note that the definition is slightly different from the usual NID used in ML
# here SoftNID = standard NIL in ML and NID is a hard version of it (using modemaps)
def NID(pW_X, pV_X, pX):
    """
    Hard NID
    :param pW_X:
    :param pV_X:
    :param pX:
    :return:
    """
    y = pW_X.argmax(axis=1)
    modemap = np.zeros_like(pW_X)
    modemap[np.arange(y.size), y] = 1
    pW_X = modemap

    y = pV_X.argmax(axis=1)
    modemap = np.zeros_like(pV_X)
    modemap[np.arange(y.size), y] = 1
    pV_X = modemap
    if len(pX.shape) == 1:
        pX = pX[:, None]
    elif pX.shape[0] == 1 and pX.shape[1] > 1:
        pX = pX.T
    pXW = pW_X * pX
    pWV = pXW.T.dot(pV_X)
    pWW = pXW.T.dot(pW_X)
    pVV = (pV_X * pX).T.dot(pV_X)
    score = 1 - MI(pWV) / (np.max([H(pWW), H(pVV)]) + PRECISION)
    return score


def softNID(pW_X, pV_X, pX):
    """
    :param pW_X:
    :param pV_X:
    :param pX:
    :return:
    """
    if len(pX.shape) == 1:
        pX = pX[:, None]
    elif pX.shape[0] == 1 and pX.shape[1] > 1:
        pX = pX.T
    pXW = pW_X * pX
    pWV = pXW.T.dot(pV_X)
    pWW = pXW.T.dot(pW_X)
    pVV = (pV_X * pX).T.dot(pV_X)
    score = 1 - MI(pWV) / (np.max([H(pWW), H(pVV)]) + PRECISION)
    return score


def get_closest_softnid(speaker, wcs, need, return_index=False):
    """ """
    gnid = np.array([softNID(speaker, wcs[i], need) for i in range(len(wcs))])
    if return_index:
        return gnid.min(), gnid.argmin()
    else:
        return gnid.min()


def get_gibson_cost(sender, listener, need):
    """
    Information cost from Gibson et al 2014
    :param sender: (N, W) npy array normalized over columns
    :param listener: (N, W) npy array normalized over rows
    :param need: (N, ) need distribution over color chips
    :return:
    """
    log_listener = np.log2(listener + PRECISION)
    tmp = np.sum(sender * log_listener, axis=1)
    return -np.sum(need.flatten() * tmp)


def bayes_listener(sender, need):
    p_w = np.sum(need.reshape(len(need), 1) * sender + PRECISION, axis=0, keepdims=True)
    listener = (need.reshape(len(need), 1) * sender) / p_w
    return listener


def lab2rgb(x):
    if len(x.shape) == 2:
        return color.lab2rgb(x[None])[0]
    elif len(x.shape) == 1:
        return color.lab2rgb(x[None, None])[0, 0]
    else:
        return color.lab2rgb(x)


def find_means(data, ib_model, major_terms=True):
    """
    Find color to use you contour plots
    :param data: npy (330, W) array
    :param ib_model: IB modoel
    :major_terms: Plot only major terms
    :return:
    """
    listener = bayes_listener(data, ib_model.pM)
    cielab = munsell[["L*", "a*", "b*"]].values
    means = []
    y = data.argmax(axis=1)
    mode_words = len(np.unique(y))
    modemap = np.zeros_like(data)
    modemap[np.arange(y.size), y] = 1
    major = np.sum(modemap, axis=0) >= 5
    if major_terms == True:
        data = data[:, major]
        data = data / np.sum(data, axis=1).reshape(330, 1)

    for w in range(data.shape[1]):
        w_probs = listener[:, w]
        if sum(w_probs) > 1e-8:
            w_ceilab = w_probs.reshape(330, 1) * cielab
            mean = np.sum(w_ceilab, axis=0, keepdims=True)  # Get average
            dist = distance_matrix(mean, cielab, p=2)
            arg_min = np.argmin(dist)
            means.append(cielab[arg_min])
    return means, data


import ib_color_naming.src.figures as ib_figures


def contour_maps(data, ib_model, fill=True, major_terms=True, plot_modemap=False):
    """

    :param data: (330, W) npy array normalized along columns
    :param ib_model: IB model used to find means based on
    :param fill: bool Fill contour plot
    :param major_terms: Plot only major terms defined as terms that are mode for at least 10 color chips
    :return:
    """

    def contour_helper(x):
        if x >= 0.9:
            return 0.9
        if x >= 0.75:
            return 0.75
        if x >= 0.5:
            return 0.5
        return 0

    if major_terms:
        y = data.argmax(axis=1)
        modemap = np.zeros_like(data)
        modemap[np.arange(y.size), y] = 1
        major = np.sum(modemap, axis=0) >= 5
        dist = data[:, major]
        data = dist / np.sum(dist, axis=1).reshape(330, 1)

    means, data = find_means(data, ib_model, major_terms=True)
    x = np.arange(1, 41)
    y = np.arange(1, 9)
    X, Y = np.meshgrid(x, y)
    for w in range(data.shape[1]):
        w_data = data[:, w]

        zdict = {
            ib_figures.cnum2ind(i + 1): contour_helper(w_data[i]) for i in range(0, 330)
        }

        Z = np.flip(
            np.array([[zdict[(i, j)] for j in range(1, 41)] for i in range(1, 9)]), 0
        )
        if plot_modemap:
            fill_colors = [np.concatenate([lab2rgb(means[w]), [1]]) for i in range(3)]
            plt.contourf(X, Y, Z, levels=[0.3, 0.75, 1], colors=fill_colors)
        else:
            fill_colors = [
                np.concatenate([lab2rgb(means[w]), [1 - (3 - i) / 8]]) for i in range(3)
            ]
            if fill:
                plt.contourf(X, Y, Z, levels=[0.3, 0.75, 1], colors=fill_colors)
            else:
                plt.contour(X, Y, Z, levels=[0.3, 0.75, 1], colors=fill_colors)
        plt.xticks([])
        plt.yticks([])


def gaussian_model(prototypes, c):
    """
    Generate Gaussian random model with precision c centered at prototypes
    :param prototypes: list of 3D cielab space vectors
    :param c: float > 0
    :return:
    """
    sender = [
        softmax([-c * (np.linalg.norm(color - x_c) ** 2) for x_c in prototypes])
        for color in cielab
    ]
    sender = np.array(sender)
    return sender


def get_gnid_ib_space(set1, set2, ib_model):
    """
    Find point closest in space w.r.t. ib objective and measure the gNID
    """
    betas = []
    gnids = []
    for encoder in set2:
        _, _, beta, _ = ib_model.fit(encoder)
        betas.append(beta)
    betas = np.array(betas)
    for encoder in set1:
        _, _, beta, _ = ib_model.fit(encoder)
        argmin = np.argmin(np.abs(betas - beta))
        gnids.append(gNID(encoder, set2[argmin], ib_model.pM.flatten()))
    return gnids


def get_closest_gnid(speaker, wcs, need, return_index=False):
    """
    Compute minimum gNID to any system in set WCS
    :param speaker: (330, W) npy array normalized along columns
    :param wcs: list of npy arrays [(N, W)] to compute closest gNID to
    :param need: need over color chips
    :param return_index: If True, return index of closest system
    :return:
    """
    gnid = np.array([gNID(speaker, wcs[i], need) for i in range(len(wcs))])
    if return_index:
        return gnid.min(), gnid.argmin()
    else:
        return gnid.min()


def get_random_model(
    wcs, ib_model, path, n_per_v=100, interval=[0.84, 2.65], threshold=0.29, seed=1
):
    """
    Generate random model. Saves random models into three pickled files,
     encoders.pkl (all models),
     attested_encoders.pkl (models with gNID to WCS below threshold) and
     unattested_encoders.pkl (models with gNID to WCS above threshold).

     Also generates pandas Dataframes for each set containing relevant information such as complexity, accuracy, inefficiency etc

     Prototypes are sampled from munsell chart without replacement

    :param wcs: List of wcs speakers. [npy(330, W)]
    :param ib_model: IB model for computing accuracy and complexity
    :param path: Save folder
    :param n_per_v: Number of systems per vocabulary size
    :param interval: Complexity interval to consider, default is WCS complexity range
    :param threshold: Threshold to split into RM_S and RM_d
    :param seed: Random seed used for sampling
    :return:
    """
    save_path = path + "random_model/"
    random_state = np.random.RandomState(seed)
    os.makedirs(save_path, exist_ok=True)

    results = []
    encoders = []
    attested_results = []
    attested_encoders = []
    unattested_results = []
    unattested_encoders = []

    params = []
    for major_terms in range(3, 11):
        i = 0
        while i < n_per_v:
            c = random_state.uniform(0.001, 0.005)
            idx = random_state.choice(330, size=major_terms, replace=False)
            means = cielab[idx]
            encoder = gaussian_model(means, c)
            complexity = ib_model.complexity(encoder)
            accuracy = ib_model.accuracy(encoder)
            deviation, gnid, beta, system = ib_model.fit(encoder)
            nid = NID(encoder, system, ib_model.pM)
            if complexity > interval[0] and complexity < interval[1]:
                i += 1
                evaluation = {
                    "Complexity": complexity,
                    "Accuracy": accuracy,
                    "Deviation": deviation,
                    "gNID": gnid,
                    "NID": nid,
                    "Beta": beta,
                }
                results.append(evaluation)
                encoders.append(encoder)
                params.append((c, means))
                wcs_gnid = get_closest_gnid(encoder, wcs, ib_model.pM)
                if wcs_gnid > threshold:
                    unattested_results.append(evaluation)
                    unattested_encoders.append(encoder)
                else:
                    attested_results.append(evaluation)
                    attested_encoders.append(encoder)

    print(f"Fraction of unattested {len(unattested_encoders) / len(encoders)}")
    df = pd.DataFrame(results)
    attested_results = pd.DataFrame(attested_results)
    unattested_results = pd.DataFrame(unattested_results)

    print(f'RM efficiency {df["Deviation"].mean()} with std {df["Deviation"].std()}')
    print(
        f'Unattested efficiency {unattested_results["Deviation"].mean()} with std {unattested_results["Deviation"].std()}'
    )
    print(
        f'Attested efficiency {attested_results["Deviation"].mean()} with std {attested_results["Deviation"].std()}'
    )

    print(
        f'Unattested range {unattested_results["Complexity"].min()} {unattested_results["Complexity"].max()}'
    )

    df.to_csv(save_path + "results.csv")
    attested_results.to_csv(save_path + "attested_results.csv")
    unattested_results.to_csv(save_path + "unattested_results.csv")

    with open(save_path + "params.pkl", "wb") as f:
        pickle.dump(params, f)

    with open(save_path + "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    with open(save_path + "unattested_encoders.pkl", "wb") as f:
        pickle.dump(unattested_encoders, f)

    with open(save_path + "attested_encoders.pkl", "wb") as f:
        pickle.dump(attested_encoders, f)


def get_vertical(max_v=10):
    """
    Hue baseline
    """
    munsell_chart = pd.read_csv("data/munsell_chart.txt", sep="\t")
    hue = munsell_chart["H"].values
    labels = np.arange(0, 41)
    hue_speakers = []
    for vocab_size in range(3, max_v + 1):
        classes = np.array_split(labels, vocab_size)
        labeling = {j: i for i in range(len(classes)) for j in classes[i]}
        # labeling[0] = 0
        speaker = np.zeros([330, vocab_size])
        for i, h in enumerate(hue):
            speaker[i, labeling[h]] = 1
        hue_speakers.append(speaker)
    return hue_speakers


def generalized_gaussian_model(prototypes, precision_matrix):
    """
    Generate Gaussian speaker using the precision matrix and words centered at prototypes
    :param prototypes:
    :param precision_matrix:
    :return:
    """
    sender = [
        softmax(
            [
                -1 / 2 * (color - x_c).T @ precision_matrix @ (color - x_c)
                for i, x_c in enumerate(prototypes)
            ]
        )
        for color in cielab
    ]
    sender = np.array(sender)
    return sender


def get_generalized_random_model(ib_model, path, n_per_v=10, seed=1):
    """
    Generate set of inefficient speakers
    :param ib_model:
    :param path:
    :param n_per_v:
    :param seed:
    :return:
    """
    save_path = path + "random_model/"
    random_state = np.random.RandomState(seed)
    os.makedirs(save_path, exist_ok=True)
    encoder_path = path + "all_encoders/"
    os.makedirs(encoder_path, exist_ok=True)

    results = []
    encoders = []
    for major_terms in range(3, 11):
        for i in range(n_per_v):
            precision_matrix = random_state.uniform(0.005, 0.01, size=[3, 3])
            idx = random_state.choice(330, size=major_terms, replace=False)
            means = cielab[idx]
            encoder = generalized_gaussian_model(means, precision_matrix)
            complexity = ib_model.complexity(encoder)
            accuracy = ib_model.accuracy(encoder)
            deviation, gnid, beta, system = ib_model.fit(encoder)
            nid = NID(encoder, system, ib_model.pM)
            evaluation = {
                "Complexity": complexity,
                "Accuracy": accuracy,
                "Deviation": deviation,
                "gNID": gnid,
                "NID": nid,
                "Beta": beta,
            }
            results.append(evaluation)
            encoders.append(encoder)
            encoder_dict = {"Model": encoder, "Eval": evaluation}
            with open(encoder_path + f"encoder_v_{major_terms}_{i}.pkl", "wb") as f:
                pickle.dump(encoder_dict, f)
    df = pd.DataFrame(results)
    df.to_csv(save_path + "results.csv")
    with open(save_path + "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    return df


def get_vertical(max_v=10):
    """
    Inefficient but learnable baseline
    :param max_v:
    :return:
    """
    hue = munsell["H"].values
    labels = np.arange(0, 41)
    hue_speakers = []
    for vocab_size in range(3, max_v + 1):
        classes = np.array_split(labels, vocab_size)
        labeling = {j: i for i in range(len(classes)) for j in classes[i]}
        # labeling[0] = 0
        speaker = np.zeros([330, vocab_size])
        for i, h in enumerate(hue):
            speaker[i, labeling[h]] = 1
        hue_speakers.append(speaker)
    return hue_speakers


import json


class NumpyEncoder(json.JSONEncoder):
    """
    Encoder for numpy to json
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


from correlation_clustering import compute_consensus_map
from sklearn.preprocessing import OneHotEncoder


def get_all_consensus_maps(encoders):
    """
    Compute consensus map for each vocab size in encoders
    """

    def mode_filter(data):
        y = data.argmax(axis=1)
        modemap = np.zeros_like(data)
        modemap[np.arange(y.size), y] = 1
        major = np.sum(modemap, axis=0) >= 10
        data = data[:, major] + 1e-20
        data = data / np.sum(data, axis=1).reshape(330, 1)
        return data

    clusters = {}
    for enc in encoders:
        enc = mode_filter(enc)
        n_words = enc.shape[1]
        if n_words not in clusters:
            clusters[n_words] = []
        clusters[n_words].append(enc)
    consensus_maps = {}
    for n_words in clusters:
        mapping = compute_consensus_map(clusters[n_words], n_words, 10)
        mapping = OneHotEncoder().fit_transform(mapping.reshape(-1, 1)).toarray()
        consensus_maps[n_words] = mapping
    return consensus_maps


def get_gnid_consensus(encoders, need, k=None, avg_centroid=False):
    def mode_filter(data):
        y = data.argmax(axis=1)
        modemap = np.zeros_like(data)
        modemap[np.arange(y.size), y] = 1
        major = np.sum(modemap, axis=0) >= 1
        data = data[:, major] + 1e-20
        data = data / np.sum(data, axis=1).reshape(330, 1)
        return data

    clusters = {}
    for enc in encoders:
        enc = mode_filter(enc)
        n_words = enc.shape[1]
        if n_words not in clusters:
            clusters[n_words] = []
        clusters[n_words].append(enc)

    gnid_consensus = {}
    for n_words in clusters:
        gnid_matrix = np.zeros([len(clusters[n_words]), len(clusters[n_words])])
        # compute pairwise gnid
        for i in range(len(clusters[n_words])):
            for j in range(len(clusters[n_words])):
                gnid_matrix[i, j] = gNID(
                    clusters[n_words][i], clusters[n_words][j], need
                )
        # save top
        if avg_centroid:
            avg_distance = np.mean(gnid_matrix, axis=1)
            # find median
            gnid_consensus[n_words] = clusters[n_words][
                np.argsort(avg_distance)[len(avg_distance) / 2]
            ]
        elif k is None:
            gnid_consensus[n_words] = clusters[n_words][
                np.argmin(np.mean(gnid_matrix, axis=1))
            ]
        else:
            # save top k
            gnid_consensus[n_words] = [
                clusters[n_words][i]
                for i in np.argsort(np.mean(gnid_matrix, axis=1))[:k]
            ]
    return gnid_consensus
