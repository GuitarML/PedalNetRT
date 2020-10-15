FROM pytorch/pytorch:latest
ADD . .

ENV MODEL="models/pedalnet.ckpt"

RUN pip install -r requirements.txt

RUN python prepare_data.py data/in.wav data/out.wav \
    python train.py \
    python test.py –-model=$MODEL \
    python convert_pedalnet_to_wavnetva.py --model=$MODEL
