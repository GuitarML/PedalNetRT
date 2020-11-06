FROM pytorch/pytorch:latest
WORKDIR /app

ADD . .
RUN python install -r requirements_docker.txt

ENV MODEL="models/pedalnet.ckpt"

CMD python prepare_data.py data/in.wav data/out.wav \
    python train.py –-model=$MODEL \
    python test.py –-model=$MODEL \
    python convert_pedalnet_to_wavenetva.py --model=$MODEL
