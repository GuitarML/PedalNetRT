import argparse
import pickle
from scipy.io import wavfile
import numpy as np
import os

def normalize(data):
    data_max = max(data)
    data_min = min(data)
    data_norm = max(data_max,abs(data_min))
    return data / data_norm

def prepare(args):
    in_rate, in_data = wavfile.read(args.in_file)
    out_rate, out_data = wavfile.read(args.out_file)
    assert in_rate == out_rate, "in_file and out_file must have same sample rate"

    # Trim the length of audio to equal the smaller wav file
    if len(in_data) > len(out_data):
      print("Trimming input audio to match output audio")
      in_data = in_data[0:len(out_data)]
    if len(out_data) > len(in_data): 
      print("Trimming output audio to match input audio")
      out_data = out_data[0:len(in_data)]

    #If stereo data, use channel 0
    if len(in_data.shape) > 1:
        print("[WARNING] Stereo data detected for in_data, only using first channel (left channel)")
        in_data = in_data[:,0]
    if len(out_data.shape) > 1:
        print("[WARNING] Stereo data detected for out_data, only using first channel (left channel)")
        out_data = out_data[:,0]    
        
    # Convert PCM16 to FP32
    if in_data.dtype == "int16":
        in_data = in_data/32767
        print("In data converted from PCM16 to FP32")
    if out_data.dtype == "int16":
        out_data = out_data/32767
        print("Out data converted from PCM16 to FP32")
    
    #normalize data
    if args.normalize == True:
        in_data = normalize(in_data)
        out_data = normalize(out_data)

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

    if not os.path.exists(os.path.dirname(args.model)):
        os.makedirs(os.path.dirname(args.model))

    pickle.dump(d, open(os.path.dirname(args.model) + "/data.pickle", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("out_file")

    parser.add_argument("--model", type=str, default="models/pedalnet/pedalnet.ckpt")
    parser.add_argument("--sample_time", type=float, default=100e-3)
    parser.add_argument("--normalize", type=bool, default=True)
    args = parser.parse_args()
    prepare(args)

