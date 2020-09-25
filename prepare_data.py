import argparse
import pickle
from scipy.io import wavfile
import numpy as np


def main(args):
    in_rate, in_data = wavfile.read(args.in_file)
    out_rate, out_data = wavfile.read(args.out_file)
    assert in_rate == out_rate, "in_file and out_file must have same sample rate"

    sample_size = int(in_rate * args.sample_time)
    length = len(in_data) - len(in_data) % sample_size

    x = in_data[:length].reshape((-1, 1, sample_size)).astype(np.float32)
    y = out_data[:length].reshape((-1, 1, sample_size)).astype(np.float32)

    split = lambda d: np.split(d, [int(len(d) * 0.6), int(len(d) * 0.8)])

    d = {}
    d["x_train"], d["x_valid"], d["x_test"] = split(x)
    d["y_train"], d["y_valid"], d["y_test"] = split(y)
    d["mean"], d["std"] = d["x_train"].mean(), d["x_train"].std()

    # standardize
    for key in "x_train", "x_valid", "x_test":
        d[key] = (d[key] - d["mean"]) / d["std"]

    pickle.dump(d, open(args.data, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("out_file")

    parser.add_argument("--data", default="data.pickle")
    parser.add_argument("--sample_time", type=float, default=100e-3)
    args = parser.parse_args()
    main(args)
