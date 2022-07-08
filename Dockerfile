FROM ubuntu
RUN apt update
RUN apt upgrade -y
RUN apt -y install python3-pip
RUN apt -y install cmake
RUN apt -y install gcc g++
RUN apt -y install python3-dev python3-numpy
RUN apt -y install libavcodec-dev libavformat-dev libswscale-dev
RUN apt -y install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
RUN apt -y install libgtk-3-dev
RUN apt -y install libpng-dev
RUN apt -y install libjpeg-dev libpng-dev libtiff-dev
RUN apt -y install libopenexr-dev
RUN apt -y install libtiff5 libtiff5-dev
RUN apt -y install libwebp-dev
RUN apt -y install build-essential
RUN apt -y install pkg-config
RUN apt -y install libopenexr-dev
RUN apt -y install libeigen3-dev
RUN apt -y install libdc1394-dev libdc1394-utils
RUN apt -y install ffmpeg
RUN apt -y install git
RUN pip install mediapipe
RUN git clone https://github.com/opencv/opencv.git
WORKDIR opencv
RUN mkdir build
WORKDIR build
RUN cmake -D WITH_GSTREAMER_0_10=ON -D WITH_1394=ON ../
RUN make
RUN make install
WORKDIR ..
WORKDIR ..
RUN pip install requests
