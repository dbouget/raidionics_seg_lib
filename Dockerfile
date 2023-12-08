# creates virtual ubuntu in docker image
FROM python:3.8-slim

# maintainer of docker file
MAINTAINER David Bouget <david.bouget@sintef.no>

# set language, format and stuff
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get -y install sudo
RUN apt-get update && apt-get install -y git

WORKDIR /workspace

# downloading source code (not necessary, mostly to run the test scripts)
RUN git clone https://github.com/dbouget/raidionics_seg_lib.git

# Python packages
RUN pip3 install --upgrade pip
RUN pip3 install -e raidionics_seg_lib/

RUN mkdir /workspace/resources

# CMD ["/bin/bash"]
ENTRYPOINT ["python3", "/workspace/raidionics_seg_lib/main.py"]




