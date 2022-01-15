FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
WORKDIR /
COPY . .
RUN pip install -r requirements-docker.txt
ENTRYPOINT ["python3", "train.py"]
