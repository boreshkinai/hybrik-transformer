# HybrIK-Transformer

This is the implementation of this paper:
https://arxiv.org/abs/2302.04774

Based on this orogonal code: https://github.com/Jeff-sjtu/HybrIK

## Register to get SMPL model
here: https://smpl.is.tue.mpg.de/register.php
https://smplify.is.tue.mpg.de/register.php

## Create workspace and clone this repository

```mkdir workspace```

```cd workspace```

```git clone git@github.com:boreshkinai/hybrik-transformer.git```

## Build docker image and launch container

Build image and start the lightweight docker container. Note that this assumes that the data for the project will be stored in the shared folder /home/pose-estimation accessible to you and other project members. 
```
cd hybrik-transformer
docker build -f Dockerfile -t hybrik_transformer:$USER .

nvidia-docker run -p 8888:8888 -p 6000-6010:6000-6010 -v ~/workspace/hybrik-transformer:/workspace/hybrik-transformer -v /home/pose-estimation/data:/workspace/hybrik-transformer/HybrIK/data -t -d --shm-size="16g" --name hybrik_transformer_$USER hybrik_transformer:$USER

```
Enter docker container and download data locally
```
docker exec -i -t hybrik_transformer_$USER  /bin/bash 
bash scripts/download_data_gcp.sh
```
Launch training session
```
cd HybrIK
./scripts/train_smpl.sh train_res34 ./configs/256x192_adam_lr1e-3-res34_smpl_3d_base_2x_mix.yaml
```
