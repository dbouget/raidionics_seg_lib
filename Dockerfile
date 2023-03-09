
# creates virtual ubuntu in docker image
FROM ubuntu:20.04

# maintainer of docker file
MAINTAINER David Bouget <david.bouget@sintef.no>

# set language, format and stuff
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# OBS: using -y is conveniently to automatically answer yes to all the questions
# installing python3 with a specific version
RUN apt-get update -y
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt update
RUN apt install python3.7 -y
RUN apt install python3.7-distutils -y
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

# installing other libraries
RUN apt-get install python3-pip -y && \
    apt-get -y install sudo
RUN apt-get install curl -y
RUN apt-get install nano -y
RUN apt-get update && apt-get install -y git

# create default user account with sudo permissions
RUN useradd -m ubuntu && echo "ubuntu:ubuntu" | chpasswd && adduser ubuntu sudo
ENV PYTHON_DIR /usr/bin/python3
RUN chown ubuntu $PYTHON_DIR -R
USER ubuntu

# downloading source code (not necessary, mostly to run the test scripts)
WORKDIR "/home/ubuntu"
RUN git clone -b v1.2.0-beta https://github.com/dbouget/raidionics_seg_lib.git

# Python packages
WORKDIR "/home/ubuntu/raidionics_seg_lib"
RUN pip3 install --upgrade pip
RUN pip3 install -e .
RUN pip3 install onnxruntime-gpu==1.12.1

# setting up a resources folder which should mirror a user folder, to "send" data/models in and "collect" the results
WORKDIR "/home/ubuntu"
USER root
RUN mkdir /home/ubuntu/resources
RUN chown -R ubuntu:ubuntu /home/ubuntu/resources
RUN chown -R ubuntu:ubuntu /home/ubuntu/raidionics_seg_lib
RUN chmod -R 777 /home/ubuntu/raidionics_seg_lib
USER ubuntu
EXPOSE 8888

# To expose the executable
ENV PATH="${PATH}:/home/ubuntu/.local/bin"

# CMD ["/bin/bash"]
ENTRYPOINT ["python3","/home/ubuntu/raidionics_seg_lib/main.py"]




