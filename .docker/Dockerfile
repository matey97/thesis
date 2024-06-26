FROM jupyter/base-notebook:python-3.9.13

ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

USER root

COPY . ${HOME}

RUN chown -R ${NB_UID} ${HOME}

RUN apt-get update && apt-get install -y build-essential pkg-config

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN rmdir work

USER ${NB_USER}