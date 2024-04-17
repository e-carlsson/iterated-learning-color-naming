import json
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from misc import get_closest_gnid, NumpyEncoder
import matplotlib.pyplot as plt
munsell_chart = pd.read_csv('data/munsell_chart.txt', sep='\t')
#sns.set(rc = {'figure.figsize':(6,3.5)})
#sns.set_theme(style='white', rc={'figure.figsize':(6,3.5)})
#sns.color_palette()
def get_vertical(max_v=10):
    '''
    Generate Hue baseline
    :param max_v:
    :return:
    '''
    hue = munsell_chart['H'].values
    labels = np.arange(0, 41)
    hue_speakers = []
    for vocab_size in range(3, max_v + 1):
        classes = np.array_split(labels, vocab_size)
        labeling = {j : i for i in range(len(classes)) for j in classes[i]}
        speaker = np.zeros([330, vocab_size])
        for i, h in enumerate(hue):
            speaker[i, labeling[h]] = 1
        hue_speakers.append(speaker)
    return hue_speakers


def initialize_experiments(rm, wcs, hue_speakers):
    # Hue speakers
    for i, hue in enumerate(hue_speakers):
        v = hue.shape[1]
        data = {
            'Model': hue,
            'Accuracy': ib_model.accuracy(hue),
            'Complexity': ib_model.complexity(hue),
            'Vocabulary Size': v,
            'Type': 'Hue',
        }
        with open(path + f'models/hue_{i}.pkl', 'wb') as f:
            pickle.dump(data, f)

    # WCS speakers
    for i, lang in enumerate(wcs):
        data = {
            'Model': lang,
            'Accuracy': ib_model.accuracy(lang),
            'Complexity': ib_model.complexity(lang),
            'Vocabulary Size': lang.shape[1],
            'Type': 'WCS'
        }
        with open(path + f'models/wcs_{i}.pkl', 'wb') as f:
            pickle.dump(data, f)

    # Random Model
    for i, lang in enumerate(rm):
        data = {
            'Model': lang,
            'Accuracy': ib_model.accuracy(lang),
            'Complexity': ib_model.complexity(lang),
            'Vocabulary Size': lang.shape[1],
            'Type': 'Random Model'
        }
        with open(path + f'models/rm_{i}.pkl', 'wb') as f:
            pickle.dump(data, f)


def load_data(exp_path, wcs, ib_model):
    experiments = glob.glob(exp_path + 'models/*.pkl')
    df = []
    for e in experiments:
        print(f'Processing {e}')
        with open(e, 'rb') as f:
            data = pickle.load(f)

        df.append(
            {
            'Accuracy': data['Accuracy'],
            'Complexity': data['Complexity'],
            'gNID': data['gNID'],
            'Type': data['Type'],
            'Vocabulary Size': data['Vocabulary Size'],
            'Distance WCS': get_closest_gnid(data['Model'], wcs, ib_model.pM)
        }
        )
    return pd.DataFrame(df)

def histogram(results, path):
    from scipy.stats import mannwhitneyu
    plt.close()
    wcs_results = results[results['Type'] == 'WCS']
    rm_results = results[results['Type'] == 'Random Model']
    hue_results = results[results['Type'] == 'Hue']
    rm_d = rm_results[rm_results['Distance WCS'] > 0.29]
    rm_s = rm_results[rm_results['Distance WCS'] <= 0.29]
    fig, ax = plt.subplots(figsize=(6, 3))
    #sns.histplot({r'$RM_d$': rm_d['gNID'], '$RM_s$': rm_s['gNID']}, common_norm=False, stat='probability',
    #             linewidth=0.2, ax=ax, color=['blue', 'green'], legend=False)
    sns.histplot({'RM': rm_results['gNID'], 'WCS': wcs_results['gNID']}, common_norm=False, stat='probability',
                 linewidth=0.2, ax=ax)
    #bins = np.linspace(0, 1, 25)
    #sns.histplot(rm_results['gNID'], common_norm=False, stat='probability', linewidth=0.2, ax=ax, label=r'RM', bins=bins)
    #sns.histplot(rm_s['gNID'], common_norm=False, stat='probability', linewidth=0.2, ax=ax, color='red', label=r'$RM_s$', bins=bins)
    #sns.histplot(wcs_results['gNID'], common_norm=False, stat='probability', linewidth=0.2, ax=ax,
    #             label=r'$RM_s$', bins=bins)
    plt.legend(['RM', 'WCS'])
    plt.xlabel('gNID Between Previous and Next Generation')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(path + 'hist_learning_wcs_rm.pdf', dpi=400)
    plt.close()

    statistics, p_value = mannwhitneyu(rm_s['gNID'].values, rm_d['gNID'].values)
    print(f'Test RM- RM-d two-sided Mann-Whitney U=:{statistics}, p {p_value}')

    statistics, p_value = mannwhitneyu(rm_s['gNID'].values, rm_d['gNID'].values, alternative='less')
    print(f'Test RM- RM-d one-sided Mann-Whitney U=:{statistics}, p {p_value}')


def tests(results, path):
    from scipy.stats import mannwhitneyu
    tests = {}
    hue = results[results['Type'] == 'Hue']
    wcs = results[results['Type'] == 'WCS']
    random = results[results['Type'] == 'Random Model']
    statistic, p_value = mannwhitneyu(hue['gNID'].values, wcs['gNID'].values, alternative='less')
    tests['Generalization error Hue < WCS'] = {'statistic': statistic, 'p-value':p_value}
    statistic, p_value = mannwhitneyu(random['gNID'].values, wcs['gNID'].values, alternative='less')
    tests['Generalization error Random < WCS'] = {'statistic': statistic, 'p-value':p_value}

    rm_d = results[results['Distance WCS'] > 0.29]
    rm_s = results[results['Distance WCS'] <= 0.29]

    print(f'rm_d median: {rm_d["gNID"].median()} and rm_s median: {rm_s["gNID"].median()}')


    statistic, p_value = mannwhitneyu(rm_d['gNID'].values, wcs['gNID'].values, alternative='less')
    tests['Generalization error RM_D < WCS'] = {'statistic': statistic, 'p-value':p_value}

    statistic, p_value = mannwhitneyu(rm_s['gNID'].values, wcs['gNID'].values, alternative='less')
    tests['Generalization error RM_S < WCS'] = {'statistic': statistic, 'p-value':p_value}

    top5 = np.quantile(results['Distance WCS'].values, q=0.05)
    bottom5 = np.quantile(results['Distance WCS'].values, q=0.95)
    rm_d = results[results['Distance WCS'] >= bottom5]
    rm_s = results[results['Distance WCS'] <= top5]

    statistic, p_value = mannwhitneyu(rm_s['gNID'].values, rm_d['gNID'].values, alternative='less')
    tests['Top < Bottom Quantile'] = {'statistic': statistic, 'p-value':p_value}



    with open(path + 'learnability_tests.json', 'w') as f:
        json.dump(tests, f, indent=4, cls=NumpyEncoder)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ini_exp', default=False, help='initialize exp. If true initialize else plot and summarize results')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import pickle
    import os
    from ib_color_naming.src.ib_naming_model import load_model

    path = 'learnability_exp/'
    ib_model = load_model()

    args = get_args()
    if args.ini_exp:
        os.makedirs(path + 'models/', exist_ok=False)
        hue_speakers = get_vertical(max_v=11)
        with open('data/wcs_encoders.pkl', 'rb') as f:
            wcs = pickle.load(f)
        # Random Model
        with open('iterated_learning_exp/random_model/encoders.pkl', 'rb') as f:
            rm = pickle.load(f)
        initialize_experiments(rm, wcs, hue_speakers)

    else:
        with open('data/wcs_encoders.pkl', 'rb') as f:
            wcs = pickle.load(f)
        ib_model = load_model()
        df = load_data(path, wcs=wcs, ib_model=ib_model)
        histogram(df, path)
        tests(df, path)
