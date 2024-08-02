import pickle
import pandas as pd
import numpy as np

from iterated_learning import iteratedLearning
from ib_color_naming.src.tools import gNID
from ib_color_naming.src.ib_naming_model import load_model

"""
Warm start experiment where NIL is initialized from a speaker and then run
"""

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", type=str, help="Model to initialize the NIL algorithm with"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    ib_model = load_model()
    args = get_args()

    with open(args.model, "rb") as f:
        data = pickle.load(f)
        model = data["Model"]
    gnid = []
    it = iteratedLearning(
        need=ib_model.pM.flatten(),
        vocabulary_size=model.shape[1],
        ib_model=ib_model,
        initial_speaker=model,
        transmission_samples=300,
        n_episodes=250,
        train_steps=1000,
    )
    encoder, t = it.run()

    print(data)
    data["Learned Model"] = encoder
    data["Log"] = it.get_log()
    with open(args.model, "wb") as f:
        pickle.dump(data, f)
