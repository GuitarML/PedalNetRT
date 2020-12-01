import pickle
import torch
from tqdm import tqdm
from scipy.io import wavfile
import argparse
import numpy as np

from model import PedalNet


def save(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.float32))


@torch.no_grad()
def predict(args):
    model = PedalNet.load_from_checkpoint(args.model)
    model.eval()
    train_data = pickle.load(open(args.train_data, "rb"))

    mean, std = train_data["mean"], train_data["std"]

    in_rate, in_data = wavfile.read(args.input)
    assert in_rate == 44100, "input data needs to be 44.1 kHz"
    sample_size = int(in_rate * args.sample_time)
    length = len(in_data) - len(in_data) % sample_size

    # split into samples
    in_data = in_data[:length].reshape((-1, 1, sample_size)).astype(np.float32)

    # standardize
    in_data = (in_data - mean) / std

    # pad each sample with previous sample
    prev_sample = np.concatenate((np.zeros_like(in_data[0:1]), in_data[:-1]), axis=0)
    pad_in_data = np.concatenate((prev_sample, in_data), axis=2)

    pred = []
    batches = pad_in_data.shape[0] // args.batch_size
    for x in tqdm(np.array_split(pad_in_data, batches)):
        pred.append(model(torch.from_numpy(x)).numpy())

    pred = np.concatenate(pred)
    pred = pred[:, :, -in_data.shape[2] :]

    save(args.output, pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/pedalnet.ckpt")
    parser.add_argument("--train_data", default="data.pickle")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_time", type=float, default=100e-3)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    predict(args)
