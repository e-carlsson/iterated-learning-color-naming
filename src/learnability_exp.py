import pickle
import numpy as np
from ib_color_naming.src.tools import gNID
from iterated_learning import iteratedLearning
from ib_color_naming.src.ib_naming_model import load_model


import argparse

"""
Run only the transmission and pre-training. Logs both the mean gNID between previous and nxt generation "Learning gNID" as well
as the value for each independent experiment "Learning Raw gNID"
"""


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

    learning_gnid = []
    for _ in range(10):
        it = iteratedLearning(
            need=ib_model.pM.flatten(),
            ib_model=ib_model,
            vocabulary_size=model.shape[1],
            initial_speaker=model,
            transmission_samples=300,
            train_steps=1000,
        )
        it.pre_train_speaker()
        encoder = it.get_speaker()
        learning_gnid.append(gNID(encoder, model, ib_model.pM))
    data["Learning gNID"] = np.mean(learning_gnid)
    data["Learning Raw gNID"] = learning_gnid

    with open(args.model, "wb") as f:
        pickle.dump(data, f)
