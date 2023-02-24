# ===========
# FIRST STAGE
# ===========
FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime as pytorch

ENV PROJECT_PATH /workspace/hybrik-transformer

RUN date
RUN apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8 && apt-get install -y git && apt-get -y install g++
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

RUN python -m pip install pip -U

# Install Jupyter
RUN conda install -y jupyter
# Install tini, which will keep the container up as a PID 1
RUN apt-get update && apt-get install -y curl grep sed dpkg && \
#    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v0.19.0/tini_0.19.0.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
# This is to support opencv and opendr
RUN apt-get update && apt-get install -y libsm6 libxrender1 libfontconfig1 libosmesa6-dev build-essential libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev

COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# ============
# SECOND STAGE
# ============
FROM pytorch

# Export port for TensorBoard
EXPOSE 6006
# Export port 8888 for jupyter
EXPOSE 8888

RUN mkdir -p -m 700 /root/.jupyter/ && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py 

ENTRYPOINT [ "/usr/bin/tini", "--" ]

CMD ["jupyter", "notebook", "--allow-root"]

RUN apt-get update && apt-get install -y wget unzip aria2 ffmpeg

RUN apt update && apt install -y apt-transport-https ca-certificates gnupg

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

WORKDIR ${PROJECT_PATH}
