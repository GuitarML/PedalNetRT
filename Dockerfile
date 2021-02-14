FROM pytorch/pytorch
WORKDIR /
COPY . .
RUN pip3 install -r requirements-docker.txt
ENTRYPOINT ["python3", "train.py", "data/in.wav", "data/out.wav"]
