FROM continuumio/miniconda3
WORKDIR /AI
COPY . /AI

RUN conda install -c conda-forge python=3.9
RUN conda install pytorch
RUN conda install torchvision
RUN conda install opencv
RUN apt-get update
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8001

RUN apt-get install ffmpeg libsm6 libxext6  -y
CMD uvicorn --host=0.0.0.0 --port 8001 main:app