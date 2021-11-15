FROM jupyter/base-notebook:python-3.9.7

USER root
RUN apt update && apt-get install -y \
    build-essential \
    python3-dev

USER jovyan
COPY ./requirements.txt requirements.txt 
RUN pip install -r requirements.txt