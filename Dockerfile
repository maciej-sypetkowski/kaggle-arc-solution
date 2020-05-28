FROM ubuntu:bionic-20200403

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y software-properties-common git cython \
        python-opencv python3-pip python3-setuptools

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# clone and patch sklearn
RUN git clone https://github.com/scikit-learn/scikit-learn.git /scikit-learn
RUN cd /scikit-learn && git checkout 5abd22f58f152a0a899f33bb22609cc085fbfdec
COPY sklearn-determinism.patch /scikit-learn
RUN cd /scikit-learn && git apply sklearn-determinism.patch

# build sklearn
RUN rm /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python
RUN cd /scikit-learn && ./setup.py install

RUN git clone https://github.com/fchollet/ARC.git /ARC && cd /ARC && git checkout 5d9066d546adf39bcb0c694d058625a5a4fedc27
RUN mkdir -p /kaggle/input
RUN ln -sf /ARC/data /kaggle/input/abstraction-and-reasoning-challenge

RUN mkdir -p /workspace/arc-solution
COPY * /workspace/arc-solution/
WORKDIR /workspace/arc-solution
