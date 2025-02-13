FROM ubuntu:22.04

# Install all needed deps
RUN apt-get update
RUN apt-get install -y python3 python3-pip python-is-python3
RUN apt-get install -y --no-install-recommends git
RUN apt-get install libgl1-mesa-glx ffmpeg libsm6 libxext6 -y

# Install python requirements
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip

# install dependencies for rendering with OpenCV
RUN apt-get install -y libglfw3
RUN apt-get install -y libglfw3-dev

RUN apt-get update && apt-get install -y cmake libgl1-mesa-dev libxinerama-dev libxcursor-dev libxrandr-dev libxi-dev ninja-build zlib1g-dev clang-12

RUN echo "alias build='bash /URDFit/experiment/mujoco_mpc/build.sh'" >> ~/.bashrc

WORKDIR /URDFit

ENV PATH $PATH:/URDFit/experiment/mujoco_mpc/build/bin/