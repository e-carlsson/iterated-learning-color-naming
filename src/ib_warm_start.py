import pandas as pd
import pickle
import glob
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings("ignore")  # Ignore matplotlib rescaling and deb warnings
import argparse
from misc import get_closest_gnid, get_gnid_ib_space, get_closest_softnid


def prepare_systems(wcs, random_model, need, save_path):
    dissim = [
        get_closest_gnid(encoder, wcs, need, return_index=False)
        for encoder in random_model
    ]

    i = 0
    for gnid, encoder in zip(dissim, random_model):
        i += 1
        data = {
            "Model": encoder,
            "gNID WCS": gnid,
        }

        with open(save_path + f"ib_{i}.pkl", "wb") as f:
            pickle.dump(data, f)

    print(f"Number of experiments {i}")


def load_data(path, wcs, ib_model):
    exp = glob.glob(path + "*.pkl")
    df = []
    models = []
    models_learned = []
    for e in exp:
        with open(e, "rb") as f:
            data = pickle.load(f)
        gnid = get_closest_gnid(
            data["Learned Model"], wcs, ib_model.pM, return_index=False
        )
        eps_learned, _, _, _ = ib_model.fit(data["Learned Model"])
        eps_model, _, _, _ = ib_model.fit(data["Model"])
        model_nid = get_closest_softnid(data["Model"], wcs, ib_model.pM)
        learned_nid = get_closest_softnid(data["Learned Model"], wcs, ib_model.pM)
        df.append(
            {
                "Vocabulary Size": data["Model"].shape[1],
                "gNID Model": data["gNID WCS"],
                "gNID Learned": gnid,
                "NID Model": model_nid,
                "NID Learned": learned_nid,
                "gNID Difference": data["gNID WCS"] - gnid,
                "NID Difference": model_nid - learned_nid,
                "Accuracy Learned": ib_model.accuracy(data["Learned Model"]),
                "Complexity Learned": ib_model.complexity(data["Learned Model"]),
                "Accuracy Initial": ib_model.accuracy(data["Model"]),
                "Complexity Initial": ib_model.complexity(data["Model"]),
                "Epsilon Learned": eps_learned,
                "Epsilon Initial": eps_model,
            }
        )
        models.append(data["Model"])
        models_learned.append(data["Learned Model"])
    gnids_learned = get_gnid_ib_space(models_learned, wcs, ib_model=ib_model)
    gnids_model = get_gnid_ib_space(models, wcs, ib_model=ib_model)
    diff = np.array(gnids_model) - np.array(gnids_learned)
    df = pd.DataFrame(df)
    df["gNID in IB space diff"] = diff

    return df, models


def plot_models(models, ib_model, path, wcs, need):
    from misc import contour_maps

    os.makedirs(path + "maps/", exist_ok=True)

    for i, m in enumerate(models):
        start = m[0]
        end = m[1]
        v = start.shape[1]
        os.makedirs(path + f"maps/model_{i}_vocab_size_{v}/", exist_ok=True)
        plt.rcParams["figure.figsize"] = (15, 5)
        plt.close()
        contour_maps(start, ib_model, fill=True)
        plt.savefig(
            path + f"maps/model_{i}_vocab_size_{v}/start_model.pdf",
            dpi=400,
            transparent=True,
        )
        plt.close()
        contour_maps(end, ib_model, fill=True)
        plt.savefig(
            path + f"maps/model_{i}_vocab_size_{v}/final_model.pdf",
            dpi=400,
            transparent=True,
        )
        plt.close()
        gnid, idx = get_closest_gnid(end, wcs, need)
        gnid = np.round(gnid, decimals=2)
        contour_maps(wcs[idx], ib_model, fill=True)
        plt.savefig(
            path + f"maps/model_{i}_vocab_size_{v}/closest_wcs_gnid_{gnid}.pdf",
            dpi=400,
            transparent=True,
        )


def ib_plot(results, ib_model, path):
    """
    Position in IB plane before and after NIl
    :param results:
    :param ib_model:
    :param path:
    :return:
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    curve = ib_model.IB_curve
    plt.plot(curve[0], curve[1], "--", color="black")
    plt.xlim([0, np.max(curve[0])])
    plt.ylim([0, ib_model.I_MU + 0.1])
    plt.fill_between(
        curve[0],
        curve[1],
        [7] * len(curve[0]),
        color="gray",
        alpha=0.3,
        label="Unachievable",
    )
    plt.xlabel("Complexity, $I(M;W)$", fontsize=12)
    plt.ylabel("Accuracy, $I(W;U)$", fontsize=12)
    sns.scatterplot(
        results,
        x="Complexity Learned",
        y="Accuracy Learned",
        ax=ax,
        label="Iterated Learning",
    )
    sns.scatterplot(
        results, x="Complexity Initial", y="Accuracy Initial", ax=ax, label="Initial"
    )

    plt.savefig(path + "ib_learned.pdf")
    plt.close()


def histogram(results, ib_model, path="warm_start_exp/"):
    """
    Histogram of gNID to WCS before and after applying NIL
    :param results:
    :param ib_model:
    :param path:
    :return:
    """
    fig, ax = plt.subplots()
    hist = results[["gNID Model", "gNID Learned"]]
    sns.histplot(hist, stat="probability", ax=ax, legend=False, linewidth=0.2, bins=25)
    plt.legend(["Before IL+C", "After IL+C"])
    plt.xlabel("Distance to WCS")
    plt.savefig(path + "warm_hist.pdf", dpi=400)
    plt.close()
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(
        results, x="gNID Difference", ax=ax, stat="probability", linewidth=0.2, bins=25
    )
    plt.xlabel("gNID Difference")
    plt.xlim([-0.35, 0.35])
    plt.tight_layout()
    # add red vertical line at x=0
    plt.axvline(x=0, color="r", linestyle="--")
    plt.savefig(path + "diff_hist.pdf", dpi=400)
    plt.close()
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(
        results, x="NID Difference", ax=ax, stat="probability", linewidth=0.2, bins=25
    )
    plt.xlabel("NID Difference")
    plt.xlim([-0.35, 0.35])
    plt.tight_layout()
    plt.savefig(path + "softnid_diff_hist.pdf", dpi=400)
    # ib_plot(df, ib_model)
    plt.close()
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(
        results,
        x="gNID in IB space diff",
        ax=ax,
        stat="probability",
        linewidth=0.2,
        bins=25,
    )
    plt.xlabel("gNID in IB space diff")
    plt.tight_layout()
    plt.savefig(path + "gnid_ib_space.pdf", dpi=400)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ini_exp",
        default=False,
        help="initialize exp. If true initialize else plot and summarize results",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    from ib_color_naming.src.ib_naming_model import load_model

    args = get_args()
    ib_model = load_model()

    with open("data/wcs_encoders.pkl", "rb") as f:
        wcs = pickle.load(f)
    if args.ini_exp:
        complexity, _ = ib_model.IB_curve
        indecies = np.argwhere((complexity > 0.84) & (complexity < 2.85)).flatten()
        encoders = [ib_model.qW_M[i] for i in indecies]

        path = "warm_start_exp_cogsci/files/"
        os.makedirs(path, exist_ok=True)
        prepare_systems(wcs, encoders, ib_model.pM, path)

    else:
        path = "warm_start_exp_cogsci/files/"
        df, models = load_data(path, wcs, ib_model)
        plt.clf()

        # Plot Histogram
        # histogram(df, ib_model, 'ib_warm_start/')
        # Plot color maps before and after learning
        # plot_models(models, ib_model, 'warm_start_exp/', wcs=wcs, need=ib_model.pM)
        histogram(df, ib_model, "ib_warm_start/")

        # Perform hypothesis test
        from scipy.stats import wilcoxon

        df.to_csv("ib_warm_start/results.csv")
        statistic, pvalue = wilcoxon(
            df["gNID Learned"].values, df["gNID Model"].values, alternative="less"
        )
        print("##### gNID #######")
        print(f"Wilcoxon test  p-value {pvalue}, statstic: {statistic}")
        print(
            f'gNID Before NIL mean: {df["gNID Model"].mean()} std: {df["gNID Model"].std()}'
        )
        print(
            f'gNID After NIL mean: {df["gNID Learned"].mean()} std: {df["gNID Learned"].std()}'
        )
        print("##### NID #######")
        statistic, pvalue = wilcoxon(
            df["NID Learned"].values, df["NID Model"].values, alternative="less"
        )
        print(f"Wilcoxon test  p-value {pvalue}, statstic: {statistic}")
        print(
            f'NID Before NIL mean: {df["NID Model"].mean()} std: {df["NID Model"].std()}'
        )
        print(
            f'NID After NIL mean: {df["NID Learned"].mean()} std: {df["NID Learned"].std()}'
        )

        print("#### efficiency #####")
        statistic, pvalue = wilcoxon(
            df["Epsilon Learned"].values,
            df["Epsilon Initial"].values,
            alternative="less",
        )
        print(f"Wilcoxon test  p-value {pvalue}, statstic: {statistic}")
        print(f'After NIL {df["Epsilon Learned"].mean()}')
        print(f'Before NIL {df["Epsilon Initial"].mean()}')

        plt.clf()
        sns.relplot(data=df, x="Vocabulary Size", y="gNID Difference")
        plt.savefig("ib_warm_start/vocab_fig.pdf")

        plt.clf()
        sns.relplot(data=df, x="Vocabulary Size", y="gNID in IB space diff")
        plt.savefig("ib_warm_start/vocab_fig.pdf")

        plt.clf()
        sns.scatterplot(data=df, x="Complexity Initial", y="gNID Difference")
        plt.savefig("ib_warm_start/comp_fig.pdf")
