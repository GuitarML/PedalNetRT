# PedalNetRT

PedalNet-RealTime is meant to be used with the 
SmartGuitarPedal, SmartGuitarAmp, and WaveNetVA plugins. 
You can train a model using this repo, then convert it to a .json 
model that can be loaded into the VST plugin. 

The following repositories are compatible with the converted .json model,
for use with real time guitar playing through a DAW plugin or stand alone app:

https://github.com/keyth72/SmartGuitarPedal

https://github.com/keyth72/SmartGuitarAmp

https://github.com/damskaggep/WaveNetVA

Re-creation of model from [Real-Time Guitar Amplifier Emulation with Deep
Learning](https://www.mdpi.com/2076-3417/10/3/766/htm)

Please see original PedalNet, without which this project would not be possible:
https://github.com/teddykoker/pedalnet

## Data

`data/ts9_test1_in_FP32.wav` - Playing from a Fender Telecaster, bridge pickup, max tone and volume<br>
`data/ts9_test1_out_FP32.wav` - Split with JHS Buffer Splitter to Ibanez TS9 Tube Screamer 
(max drive, mid tone and volume).<br>
`models/ts9_epoch=1362.ckpt` - Pretrained model weights


## Usage

**Run effect on .wav file**:
Must be single channel, 44.1 kHz, FP32 wav data (not int16)
```bash
# must be same data used to train
python prepare_data.py data/ts9_test1_in_FP32.wav data/ts9_test1_out_FP32.wav 

# specify input file and desired output file
python predict.py my_input_guitar.wav my_output.wav 

# if you trained you own model you can pass --model flag
# with path to .ckpt
```

**Train**:
```bash
python prepare_data.py data/ts9_test1_in_FP32.wav data/ts9_test1_out_FP32.wav  # or use your own!
python train.py 
python train.py --gpus "0,1"  # for multiple gpus
python train.py -h  # help (see for other hyperparameters)
```

**Test**:
```bash
python test.py # test pretrained model
python test.py --model lightning_logs/version_{X}/epoch={EPOCH}.ckpt  # test trained model
```
Creates files `y_test.wav`, `y_pred.wav`, and `x_test.wav`, for the ground truth
output, predicted output, and input signal respectively.


**Model Conversion**:

The .ckpt model must be converted to a .json model to run in the plugin.
Usage:

	python convert_pedalnet_to_wavnetva.py --model=your_trained_model.ckpt

Generates a file named "converted_model.json" that can be loaded into the VST plugin.

**Analysis**:

You can also use "plot_wav.py" to evaluate the trained PedalNet model. By 
default, this will analyze the three .wav files from the test.py output. It 
will output analysis plots and calculate the error to signal ratio. 

Usage (after running "python test.py --model=your_model.ckpt"):

	python plot_wav.py

![app](https://github.com/keyth72/pedalnet/blob/master/figures/example_plot.png)

## Training Info
Differences from the original PedalNet (to make compatible with WaveNet plugin):
1. Uses a custom Causal Padding mode not available in PyTorch.
2. Uses a single conv1d layer for both sigm and tanh calculations, instead of 
   two separate layers.
3. Adds a conv1d input layer.
4. Requires float32 .wav files for training (instead of int16).

Helpful tips on training models:
1. Wav files should be 3 - 4 minutes long, and contain a variety of
   chords, individual notes, and playing techniques to get a full spectrum
   of data for the model to "learn" from.
2. A buffer splitter was used with pedals to obtain a pure guitar signal
   and post effect signal.
3. Obtaining sample data from an amp can be done by splitting off the original 
   signal, with the post amp signal coming from a microphone (I used a SM57).
   Keep in mind that this captures the dynamic response of the mic and cabinet.
   In the original research the sound was captured directly from within the amp
   circuit to have a "pure" amp signal.
4. Generally speaking, the more distorted the effect/amp, the more difficult it
   is to train. Experiment with different hyperparameters for each target 
   hardware. I found that a model with only 5 channels was able to sufficiently
   model some effects, and this reduces the model size and allows the plugin 
   to use less processing power.
5. When recording samples, try to maximize the volume levels without clipping.
   The levels you train the model at will be reproduced by the plugin. Also try 
   to make the pre effect and post effect wav samples equal in volume levels. 
   Even though the actual amp or effect may raise the level significantly, this isn't 
   necessarily desirable in the end plugin.
   
   
 Note: Added an experimental Google Colab notebook to train pedalnet models on TPU's.
       Upload "colab_TPU_training.ipynb" in Google Colab, and upload this pedalnet
       repository to your Google Drive to use.



