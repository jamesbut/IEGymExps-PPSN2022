FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3.9 python3.9-dev python3-pip swig

WORKDIR /app
ADD . /app

RUN cd lib/deap && python3.9 -m pip install -e .
RUN cd lib/gym && python3.9 -m pip install -e .
RUN python3.9 -m pip --default-timeout=1000 install -r requirements.txt

ENTRYPOINT ["python3.9"]
