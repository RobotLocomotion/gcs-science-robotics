# -*- mode: dockerfile -*-
# vi: set ft=dockerfile :

# git clone -b gcs-science-robotics git@github.com:wrangelvid/drake.git
# mkdir drake-build
# cd drake-build
# cmake -DWITH_MOSEK=ON ../drake
# make -j
# Then copy this Dockerfile into the drake-build folder
# docker build -f Dockerfile -t wrangelvid/drake:gcs-science-robotics ./install
#
# docker login
# docker push wrangelvid/drake:gcs-science-robotics

FROM ubuntu:jammy
ARG DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash
# Open ports for meshcat:
EXPOSE 7000-7099/tcp
EXPOSE 8888/tcp

LABEL org.opencontainers.image.authors="David von Wrangel"
LABEL org.opencontainers.image.description="Drake with sampling based planning for the gcs paper"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"
LABEL org.opencontainers.image.source="https://github.com/wrangelvid/drake/tree/gcs-science-robotics"
LABEL org.opencontainers.image.vendor="Massachusetts Institute of Technology"

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install curl and useful transport
RUN apt-get update && yes "Y" \
      | apt-get install --no-install-recommends curl apt-transport-https
RUN apt install -y tzdata
COPY . /opt/drake/

# Install drake prereqs.
RUN apt-get update \
  && yes "Y" | bash /opt/drake/share/drake/setup/install_prereqs

RUN apt install python3-pip -y
RUN apt install python-is-python3

# Install deepnote dependencies.
RUN apt-get update \
  && yes "Y" | bash /opt/drake/share/drake/setup/deepnote/install_nginx
RUN apt-get update \
  && yes "Y" | bash /opt/drake/share/drake/setup/deepnote/install_xvfb

RUN apt-get -q update && apt-get -q install -y --no-install-recommends git nginx-light xvfb && apt-get -q clean
RUN pip3 install lxml matplotlib networkx numpy pandas pydot scipy
RUN pip3 install -I ipywidgets==7.5.1
RUN pip3 install jupyter_server
RUN pip3 install --upgrade notebook

# Install GCS repository.
ENV DISPLAY=:1
ENV MOSEKLM_LICENSE_FILE="/tmp/mosek.lic"
RUN git clone https://github.com/RobotLocomotion/gcs-science-robotics.git
ENV PYTHONPATH=/gcs-science-robotics:${PYTHONPATH}

# Source environment of Drake installed to /drake
ENV PYTHONPATH=/opt/drake/lib/python3.10/site-packages:${PYTHONPATH}
