import json

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
from ib_color_naming.src.tools import gNID
import argparse
from misc import (
    NID,
    softNID,
    NumpyEncoder,
    get_closest_gnid,
    contour_maps,
    get_conditional_entropy,
    get_entropy,
    get_soft_convexity,
    get_convexity,
)
from ib_color_naming.src.ib_naming_model import load_model


def load_data(exp_path, ib_model=None):
    files = f"{exp_path}files/*.pkl"
    experiments = glob.glob(files)
    df = []
    encoders = []
    for e in experiments:
        with open(e, "rb") as f:
            # print(f'Processing {e}')
            data = pickle.load(f)
            encoder = data[0]
            results = data[1]
            results["dc"] = get_convexity(encoder)
            results["soft-dc"] = get_soft_convexity(encoder, ib_model.pM.flatten())
            if ib_model is not None:
                # H(W) and H(W|C)
                results["Entropy"] = get_entropy(encoder, ib_model.pM.flatten())
                results["Normalized Entropy"] = results["Entropy"] / np.log2(
                    encoder.shape[1]
                )
                results["Conditional Entropy"] = get_conditional_entropy(
                    encoder, ib_model.pM.flatten()
                )
                results["Normalized Complexity"] = results["Complexity"] / np.log2(
                    encoder.shape[1]
                )
            encoders.append(encoder)
            df.append(results)
    return pd.DataFrame(df), encoders


def plot_ib(
    results, wcs_results, rm_results, hue_results, ib_model, save_path, exp_type="IL+C"
):
    """
    Plot IB with WCS and RM + IB with iterated learning
    :param results: Iterated learning results as DataFrame
    :param wcs_results: WCS results as DataFrame
    :param rm_results: RM results as DataFrame
    :param: hue_results: Results for Hue baseline
    :param ib_model:
    :param save_path:
    :return:
    """

    # IB with WCS, hue, RM
    fig, ax = plt.subplots(figsize=(7, 4))
    curve = ib_model.IB_curve
    plt.plot(curve[0], curve[1], "--", color="black")
    plt.fill_between(
        curve[0],
        curve[1],
        [7] * len(curve[0]),
        color="gray",
        alpha=0.3,
        label="Unachievable",
    )
    plt.xlabel("Complexity", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlim([0.0, 4])
    plt.ylim([0, 4])
    sns.scatterplot(
        rm_results, x="Complexity", y="Accuracy", ax=ax, label="RM", alpha=1
    )
    sns.scatterplot(
        wcs_results, x="Complexity", y="Accuracy", ax=ax, label="WCS", alpha=0.5
    )
    sns.scatterplot(
        hue_results,
        x="Complexity",
        y="Accuracy",
        ax=ax,
        label="Hue baseline",
        alpha=0.5,
    )

    plt.legend()
    plt.savefig(save_path + "wcs_rm_hue_ib.pdf", dpi=400)
    plt.close()

    # Iterated learning IB
    fig, ax = plt.subplots(figsize=(7, 4))
    curve = ib_model.IB_curve
    plt.plot(curve[0], curve[1], "--", color="black")
    plt.xlim([0.0, 4])
    plt.ylim([0, 4])
    plt.fill_between(
        curve[0],
        curve[1],
        [7] * len(curve[0]),
        color="gray",
        alpha=0.3,
        label="Unachievable",
    )
    plt.xlabel("Complexity", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    sns.scatterplot(
        rm_results, x="Complexity", y="Accuracy", ax=ax, label="RM", alpha=1
    )
    sns.scatterplot(
        results, x="Complexity", y="Accuracy", ax=ax, label=exp_type, alpha=0.5
    )

    plt.legend()
    plt.savefig(save_path + "agent_ib.pdf", dpi=400)
    plt.close()

    # Iterated learning IB
    fig, ax = plt.subplots(figsize=(7, 4))
    curve = ib_model.IB_curve
    plt.plot(curve[0], curve[1], "--", color="black")
    plt.xlim([0.8, 3])
    plt.ylim([0, 3])
    plt.fill_between(
        curve[0],
        curve[1],
        [7] * len(curve[0]),
        color="gray",
        alpha=0.3,
        label="Unachievable",
    )
    plt.xlabel("Complexity", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    sns.scatterplot(
        rm_results, x="Complexity", y="Accuracy", ax=ax, label="RM", alpha=0.5
    )
    sns.scatterplot(results, x="Complexity", y="Accuracy", ax=ax, label="IL", alpha=1)
    sns.scatterplot(
        hue_results, x="Complexity", y="Accuracy", ax=ax, label="Hue", alpha=1
    )

    plt.legend()
    plt.savefig(save_path + "iterated_ib_with_hue.pdf", dpi=400)
    plt.close()

    # Exp against WCS
    fig, ax = plt.subplots(figsize=(7, 4))
    curve = ib_model.IB_curve
    plt.plot(curve[0], curve[1], "--", color="black", zorder=-1)
    plt.xlim([0.0, 4])
    plt.ylim([0, 4])
    plt.fill_between(
        curve[0],
        curve[1],
        [7] * len(curve[0]),
        color="gray",
        alpha=0.3,
        label="Unachievable",
    )
    plt.xlabel("Complexity", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    sns.scatterplot(
        wcs_results, x="Complexity", y="Accuracy", ax=ax, label="WCS", alpha=1
    )
    sns.scatterplot(
        results, x="Complexity", y="Accuracy", ax=ax, label=exp_type, alpha=0.5
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + f"wcs_{exp_type}_ib.pdf", dpi=400)
    plt.close()

    # Iterated learning IB
    fig, ax = plt.subplots(figsize=(7, 4))
    curve = ib_model.IB_curve
    plt.plot(curve[0], curve[1], "--", color="black")
    plt.xlim([0, 7])
    plt.ylim([0, 5])
    plt.fill_between(
        curve[0],
        curve[1],
        [7] * len(curve[0]),
        color="gray",
        alpha=0.3,
        label="Unachievable",
    )
    plt.xlabel("Complexity", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    sns.scatterplot(rm_results, x="Complexity", y="Accuracy", ax=ax, label="RM")
    sns.scatterplot(
        wcs_results,
        x="Complexity",
        y="Accuracy",
        ax=ax,
        label="WCS",
        alpha=0.75,
        marker="o",
    )
    plt.legend()

    plt.savefig(save_path + "rm_wcs_ib.pdf", dpi=400)
    plt.close()

    # Plot H(W) against Complexity
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.xlim([0, 4])
    plt.ylim([0, 5])
    plt.xlabel("Complexity", fontsize=12)
    plt.ylabel("H(W)", fontsize=12)
    sns.scatterplot(rm_results, x="Complexity", y="Entropy", ax=ax, label="RM")
    sns.scatterplot(
        wcs_results,
        x="Complexity",
        y="Entropy",
        ax=ax,
        label="WCS",
        alpha=0.75,
        marker="o",
    )
    sns.scatterplot(
        results, x="Complexity", y="Entropy", ax=ax, label=exp_type, alpha=0.75
    )
    plt.legend()

    plt.savefig(save_path + "HW.pdf", dpi=400)
    plt.close()

    # Plot H(W|C) against Complexity
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.xlim([0, 4])
    plt.ylim([0, 5])
    plt.xlabel("Complexity", fontsize=12)
    plt.ylabel("H(W|C)", fontsize=12)
    sns.scatterplot(
        rm_results, x="Complexity", y="Conditional Entropy", ax=ax, label="RM"
    )
    sns.scatterplot(
        wcs_results,
        x="Complexity",
        y="Conditional Entropy",
        ax=ax,
        label="WCS",
        alpha=0.75,
        marker="o",
    )
    sns.scatterplot(
        results,
        x="Complexity",
        y="Conditional Entropy",
        ax=ax,
        label=exp_type,
        alpha=0.75,
    )
    plt.legend()

    plt.savefig(save_path + "HWC.pdf", dpi=400)
    plt.close()

    # Plot H(W) against H(W|C)
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.xlim([0, 5])
    plt.ylim([0, 5])
    plt.xlabel("H(W)", fontsize=12)
    plt.ylabel("H(W|C)", fontsize=12)
    sns.scatterplot(rm_results, x="Entropy", y="Conditional Entropy", ax=ax, label="RM")
    sns.scatterplot(
        wcs_results,
        x="Entropy",
        y="Conditional Entropy",
        ax=ax,
        label="WCS",
        alpha=0.75,
        marker="o",
    )
    sns.scatterplot(
        results, x="Entropy", y="Conditional Entropy", ax=ax, label=exp_type, alpha=0.75
    )
    plt.legend()
    plt.savefig(save_path + "HWHWC.pdf", dpi=400)
    plt.close()

    # Plot Normalized Entropy Against Complexity
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.xlim([0, 4])
    plt.ylim([0, 1])
    plt.xlabel("Complexity", fontsize=12)
    plt.ylabel("H(W)/log2(V)", fontsize=12)
    sns.scatterplot(
        rm_results, x="Complexity", y="Normalized Entropy", ax=ax, label="RM"
    )
    sns.scatterplot(
        wcs_results,
        x="Complexity",
        y="Normalized Entropy",
        ax=ax,
        label="WCS",
        alpha=0.75,
        marker="o",
    )
    sns.scatterplot(
        results,
        x="Complexity",
        y="Normalized Entropy",
        ax=ax,
        label=exp_type,
        alpha=0.75,
    )
    plt.legend()
    plt.savefig(save_path + "normed_entropy.pdf", dpi=400)
    plt.close()

    # Plot Normalized Complexity Against Complexity
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.xlim([0, 4])
    plt.ylim([0, 1])
    plt.xlabel("Complexity", fontsize=12)
    plt.ylabel("I(W;C)/log2(V)", fontsize=12)
    sns.scatterplot(
        rm_results, x="Complexity", y="Normalized Complexity", ax=ax, label="RM"
    )
    # sns.scatterplot(wcs_results, x='Complexity', y='Normalized Complexity', ax=ax, label='WCS', alpha=0.75, marker='o')
    sns.scatterplot(
        results,
        x="Complexity",
        y="Normalized Complexity",
        ax=ax,
        label=exp_type,
        alpha=0.75,
    )
    plt.legend()
    plt.savefig(save_path + "normed_complexity.pdf", dpi=400)
    plt.close()

    # plot histogram of complexity for IL and WCS
    fig, ax = plt.subplots(figsize=(7, 0.5))
    plt.xlim([0, 4])
    plt.ylim([0, 0.25])
    sns.histplot(
        wcs_results,
        x="Complexity",
        ax=ax,
        stat="probability",
        linewidth=0.2,
        bins=20,
        label="WCS",
    )
    sns.histplot(
        results,
        x="Complexity",
        ax=ax,
        stat="probability",
        linewidth=0.2,
        bins=20,
        label=exp_type,
    )
    # plt.legend()
    ax.yaxis.set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.savefig(save_path + "complexity_hist.pdf")
    plt.close()

    # print convexity results
    print(f"####### Convexity #######")
    print(f'{exp_label} convexity: {results["dc"].mean()} std {results["dc"].std()}')


def random_model_metrics(wcs_df, rm_df, rm_s, rm_d, ib_model, save_path):
    """
        Compute deviation  from optimality for RM, RM_S and RM_D
    :param wcs_df:
    :param rm_df:
    :param rm_s:
    :param rm_d:
    :param ib_model:
    :param save_path:
    :return:
    """
    from scipy.stats import mannwhitneyu

    metrics = {}
    metrics["RM Deviation Mean"] = rm_df["Deviation"].mean()
    metrics["RM Deviation Median"] = rm_df["Deviation"].median()
    metrics["RM Deviation Std"] = rm_df["Deviation"].std()

    rms_df = []
    for rm in rm_s:
        deviation, _, _, _ = ib_model.fit(rm)
        rms_df.append({"Deviation": deviation})
    rms_df = pd.DataFrame(rms_df)

    rmd_df = []
    for rm in rm_d:
        deviation, _, _, _ = ib_model.fit(rm)
        rmd_df.append({"Deviation": deviation})
    rmd_df = pd.DataFrame(rmd_df)

    metrics["RM-S Deviation Mean"] = rms_df["Deviation"].mean()
    metrics["RM-S Deviation Median"] = rms_df["Deviation"].median()
    metrics["RM-S Deviation Std"] = rms_df["Deviation"].std()
    metrics["RM-S size"] = len(rms_df["Deviation"])

    metrics["RM-D Deviation Mean"] = rmd_df["Deviation"].mean()
    metrics["RM-D Deviation Median"] = rmd_df["Deviation"].median()
    metrics["RM-D Deviation Std"] = rmd_df["Deviation"].std()
    metrics["RM-D size"] = len(rmd_df["Deviation"])

    metrics["WCS size"] = len(wcs_results)

    # Test RM vs WCS
    statistic, p_value = mannwhitneyu(
        rm_df["Deviation"].values, wcs_df["Deviation"].values, alternative="less"
    )
    metrics["Deviation RM < WCS with Mann-Whitney U"] = {
        "statistic": statistic,
        "p-value": p_value,
    }

    # Test RM-S vs RM-S
    statistic, p_value = mannwhitneyu(
        rmd_df["Deviation"].values, rms_df["Deviation"].values, alternative="less"
    )
    metrics["Deviation RM-D < RM-S with Mann-Whitney U"] = {
        "statistic": statistic,
        "p-value": p_value,
    }

    # Test RM-D vs WCS
    statistic, p_value = mannwhitneyu(
        rmd_df["Deviation"].values, wcs_df["Deviation"].values, alternative="less"
    )
    metrics["Deviation RM-D < WCS with Mann-Whitney U"] = {
        "statistic": statistic,
        "p-value": p_value,
    }

    with open(save_path + "random_model_metrics_and_tests.json", "w") as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)


def gnid_to_optimal(wcs_df, rm_df, rms_df, rmd_df, save_path):
    from scipy.stats import mannwhitneyu

    metrics = {}
    metrics["WCS gNID Mean"] = wcs_df["gNID"].mean()
    metrics["WCS gNID Std"] = wcs_df["gNID"].std()

    metrics["RM gNID Mean"] = rm_df["gNID"].mean()
    metrics["RM gNID Std"] = rm_df["gNID"].std()

    metrics["RM-S gNID Mean"] = rms_df["gNID"].mean()
    metrics["RM-S gNID Std"] = rms_df["gNID"].std()

    metrics["RM-D gNID Mean"] = rmd_df["gNID"].mean()
    metrics["RM-D gNID Std"] = rmd_df["gNID"].std()

    statistic, p_value = mannwhitneyu(wcs_df["gNID"], rm_df["gNID"], alternative="less")
    metrics["gNID to optimal WCS < RM"] = {"statistic": statistic, "p value": p_value}

    with open(save_path + "gnid2optimal.json", "w") as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)


def il_metrics(
    df,
    il_systems,
    wcs,
    random_model,
    ib_model,
    save_path,
    rm_s=None,
    rm_d=None,
    kmeans_systems=None,
):
    """
        Compute various metrics for the iterated learning systems and dump in a file "metrics_and_tests.json"
    :param df: iterated learning results as dataframe
    :param il_systems:
    :param wcs:
    :param random_model:
    :param ib_model:
    :param save_path:
    :param rm_s:
    :param rm_d:
    :return:
    """
    from scipy.stats import wilcoxon, spearmanr, mannwhitneyu

    metrics = {}
    print(
        f'Iterated learning deviation: {df["Deviation"].mean()} std {df["Deviation"].std()}'
    )
    metrics["Deviation Mean"] = df["Deviation"].mean()
    metrics["Deviation Median"] = df["Deviation"].median()
    metrics["Deviation Std"] = df["Deviation"].std()
    metrics["Iterated Learning Size"] = len(df)
    at_zero = df[df["Complexity"] < 1e-2]
    metrics["Iterated Learning Precent at 0"] = len(at_zero) / len(df)
    print(f"Iterated learning Precent at 0: {len(at_zero) / len(df)}")

    gnid_to_rm = [
        get_closest_gnid(encoder, random_model, ib_model.pM) for encoder in il_systems
    ]
    gnid_to_wcs = [
        get_closest_gnid(encoder, wcs, ib_model.pM) for encoder in il_systems
    ]
    df["gNID to WCS"] = gnid_to_wcs
    df["gNID to RM"] = gnid_to_rm
    print(
        f"Iterated learning gNID to WCS: {np.mean(gnid_to_wcs)} std {np.std(gnid_to_wcs)}"
    )
    metrics["gNID IL-WCS Mean"] = np.mean(gnid_to_wcs)
    metrics["gNID IL-WCS Std"] = np.std(gnid_to_wcs)
    metrics["gNID IL-RM Mean"] = np.mean(gnid_to_rm)
    metrics["gNID IL-RM Std"] = np.std(gnid_to_rm)

    if rm_s is not None:
        gnid_to_rms = [
            get_closest_gnid(encoder, rm_s, ib_model.pM) for encoder in il_systems
        ]
        print(
            f"Iterated learning gNID to RM-S: {np.mean(gnid_to_rms)} std {np.std(gnid_to_rms)}"
        )
        metrics["gNID IL-RM_S Mean"] = np.mean(gnid_to_rms)
        metrics["gNID IL-RM_S Std"] = np.std(gnid_to_rms)

        statistic, p_value = wilcoxon(gnid_to_wcs, gnid_to_rms, alternative="greater")
        wcs_rmd_test = {"statistic": statistic, "p-value": p_value}
        metrics["Wilcoxon IL to WCS > IL to RM-S"] = wcs_rmd_test

    if rm_d is not None:
        gnid_to_rmd = [
            get_closest_gnid(encoder, rm_d, ib_model.pM) for encoder in il_systems
        ]
        print(
            f"Iterated learning gNID to RM-D: {np.mean(gnid_to_rmd)} std {np.std(gnid_to_rmd)}"
        )
        metrics["gNID IL-RM_D Mean"] = np.mean(gnid_to_rmd)
        metrics["gNID IL-RM_D Std"] = np.std(gnid_to_rmd)

        statistic, p_value = wilcoxon(gnid_to_wcs, gnid_to_rmd, alternative="less")
        wcs_rmd_test = {"statistic": statistic, "p-value": p_value}
        metrics["Wilcoxon IL to WCS < IL to RM-D"] = wcs_rmd_test

    if kmeans_systems is not None:
        # IL to Kmeans
        gnid_to_kmeans = [
            get_closest_gnid(encoder, kmeans_systems, ib_model.pM)
            for encoder in il_systems
        ]
        print(
            f"Iterated learning gNID to KMEANS: {np.mean(gnid_to_kmeans)} std {np.std(gnid_to_kmeans)}"
        )
        metrics["gNID IL-KMEANS Mean"] = np.mean(gnid_to_kmeans)
        metrics["gNID IL-KMEANS Std"] = np.std(gnid_to_kmeans)
        # WCS to kmeans
        gnid_to_kmeans = [
            get_closest_gnid(encoder, kmeans_systems, ib_model.pM) for encoder in wcs
        ]
        print(
            f"WCS gNID to KMEANS: {np.mean(gnid_to_kmeans)} std {np.std(gnid_to_kmeans)}"
        )
        metrics["gNID WCS-KMEANS Mean"] = np.mean(gnid_to_kmeans)
        metrics["gNID WCS-KMEANS Std"] = np.std(gnid_to_kmeans)
        # RM-D to kmeans
        gnid_to_kmeans = [
            get_closest_gnid(encoder, kmeans_systems, ib_model.pM) for encoder in rm_d
        ]
        print(
            f"RM-D gNID to KMEANS: {np.mean(gnid_to_kmeans)} std {np.std(gnid_to_kmeans)}"
        )
        metrics["gNID RM-D-KMEANS Mean"] = np.mean(gnid_to_kmeans)
        metrics["gNID RM-D-KMEANS Std"] = np.std(gnid_to_kmeans)

    gnid_rm_wcs = [get_closest_gnid(rm, wcs, ib_model.pM) for rm in random_model]
    plt.close()
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(
        {"Random Model": gnid_rm_wcs, "Iterated Learning": gnid_to_wcs},
        common_norm=False,
        stat="probability",
        ax=ax,
        linewidth=0.2,
    )
    plt.legend(["RM", "IL+C"])
    plt.xlabel("gNID to WCS")
    plt.ylim(0, 0.2)
    plt.tight_layout()
    plt.savefig(save_path + "distribution_distance_wcs.pdf", dpi=400)
    plt.close()

    fig, ax = plt.subplots(2, 1)
    import itertools

    bins = np.linspace(0, 1, 20)
    palette = itertools.cycle(sns.color_palette())
    sns.histplot(
        gnid_rm_wcs,
        ax=ax[0],
        stat="probability",
        linewidth=0.2,
        color=next(palette),
        bins=bins,
    )
    ax[0].set_xlabel(None)
    ax[0].set_ylabel(None)
    # ax[0].set_xticklabels([])
    # ax[0].set_yticklabels([])
    ax[0].set_ylim(0, 0.6)
    ax[0].set_xlim(0, 0.7)
    ax[0].legend(["RM"])
    sns.histplot(
        gnid_to_wcs,
        ax=ax[1],
        stat="probability",
        linewidth=0.2,
        color=next(palette),
        bins=bins,
    )
    ax[1].set_xlabel("gNID to WCS")
    ax[1].legend(["IL+C"])
    ax[1].set_ylim(0, 0.6)
    ax[1].set_xlim(0, 0.7)
    plt.savefig(save_path + "distribution_distance_wcs_subplots.pdf", dpi=400)

    plt.close()
    sns.histplot(gnid_rm_wcs)
    plt.savefig(save_path + "rm_gnid_dist.pdf", dpi=400)
    statistic, p_value = mannwhitneyu(gnid_to_wcs, gnid_rm_wcs, alternative="less")
    metrics["Mann-Whitney U IL < RM similarity to WCS"] = {
        "statistic": statistic,
        "p-value": p_value,
    }

    # Check RM correlation
    gnid_rm_il_systems = [
        get_closest_gnid(rm, il_systems, ib_model.pM) for rm in random_model
    ]

    spearman_corr, p_value = spearmanr(
        np.array(gnid_rm_il_systems), np.array(gnid_rm_wcs)
    )
    metrics["RM Spearman Correlation"] = {
        "Correlation": spearman_corr,
        "p-value": p_value,
    }

    with open(save_path + "itterated_learning_metrics_and_tests.json", "w") as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)

    return df


def plot_maps(maps, ib_model, path):
    os.makedirs(path, exist_ok=True)
    plt.rcParams["figure.figsize"] = (15, 5)
    for (
        i,
        system,
    ) in enumerate(maps):
        plt.close()
        v = system.shape[1]
        contour_maps(system, ib_model, fill=True)
        plt.savefig(path + f"model_{i}_vocab_size_{v}.pdf", dpi=400, transparent=True)
        plt.close()


def get_results(wcs, ib_model):
    """
    Reproduce results from Zaslavsky et al. (2018)
    :param wcs:
    :param ib_model:
    :return:
    """
    wcs_results = []
    for lang in wcs:
        deviation, gnid, beta, system = ib_model.fit(lang)
        nid = NID(lang, system, ib_model.pM)
        wcs_results.append(
            {
                "Accuracy": ib_model.accuracy(lang),
                "Complexity": ib_model.complexity(lang),
                "Deviation": deviation,
                "gNID": gnid,
                "NID": nid,
                "Entropy": get_entropy(lang, ib_model.pM.flatten()),
                "Conditional Entropy": get_conditional_entropy(
                    lang, ib_model.pM.flatten()
                ),
                "Normalized Entropy": get_entropy(lang, ib_model.pM.flatten())
                / np.log2(lang.shape[1]),
                "Normalized Complexity": ib_model.complexity(lang)
                / np.log2(lang.shape[1]),
            }
        )

    return pd.DataFrame(wcs_results)


from misc import get_vertical

if __name__ == "__main__":
    ib_model = load_model()
    path = "iterated_learning_exp_cogsci/"
    exp_type = "NIL/"
    exp_label = "IL+C"
    with open("data/wcs_encoders.pkl", "rb") as f:
        wcs = pickle.load(f)
    with open(path + "random_model/encoders.pkl", "rb") as f:
        random_model = pickle.load(f)
    with open(path + "random_model/unattested_encoders.pkl", "rb") as f:
        rm_d = pickle.load(f)
    with open(path + "random_model/attested_encoders.pkl", "rb") as f:
        rm_s = pickle.load(f)
    wcs_results = get_results(wcs, ib_model)
    rms_df = get_results(rm_s, ib_model)
    rmd_df = get_results(rm_d, ib_model)
    # rm_results = pd.read_csv(path + 'random_model/results.csv')
    rm_results = get_results(random_model, ib_model)
    path = path + exp_type
    df, il_systems = load_data(path, ib_model=ib_model)
    with open(path + "encoders.pkl", "wb") as f:
        pickle.dump(il_systems, f)
    df.to_csv(path + "learning_results.csv")
    hue_baseline = get_vertical(max_v=10)
    hue_results = get_results(hue_baseline, ib_model)
    # load kmeans systems
    with open("kmeans_systems/kmeans_systems.pkl", "rb") as f:
        kmeans_systems = pickle.load(f)
    kmeans_results = get_results(kmeans_systems, ib_model)
    kmeans_results.to_csv("kmeans_systems/kmeans_results.csv")

    plot_ib(
        results=df,
        wcs_results=wcs_results,
        rm_results=rm_results,
        ib_model=ib_model,
        hue_results=hue_results,
        save_path=path,
        exp_type=exp_label,
    )
    gnid_to_optimal(
        wcs_df=wcs_results,
        rm_df=rm_results,
        rms_df=rms_df,
        rmd_df=rmd_df,
        save_path=path,
    )
    df = il_metrics(
        df=df,
        il_systems=il_systems,
        wcs=wcs,
        random_model=random_model,
        ib_model=ib_model,
        save_path=path,
        rm_s=rm_s,
        rm_d=rm_d,
        kmeans_systems=kmeans_systems,
    )
    df.to_csv(path + "learning_results.csv")
    random_model_metrics(
        wcs_df=wcs_results,
        rm_df=rm_results,
        rm_s=rm_s,
        rm_d=rm_d,
        ib_model=ib_model,
        save_path=path,
    )
    # plot_maps(il_systems, ib_model, path=path +'il_maps/')
