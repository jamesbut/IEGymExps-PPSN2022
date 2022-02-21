FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python-opengl \
    swig \
    xserver-xephyr \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


ADD requirements.txt /app/requirements.txt
WORKDIR /app
RUN python3.9 -m pip --default-timeout=1000 install -r requirements.txt

ADD lib /app/lib
RUN cd lib/deap && python3.9 -m pip install -e .
RUN cd lib/gym && python3.9 -m pip install -e .

ENTRYPOINT ["python3.9"]
