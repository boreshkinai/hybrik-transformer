# 3D Human Pose and Shape Estimation via HybrIK-Transformer

This is the implementation of this paper:
https://arxiv.org/abs/2302.04774

Based on this original code: https://github.com/Jeff-sjtu/HybrIK

## Register to get SMPL model
here: https://smpl.is.tue.mpg.de/register.php
https://smplify.is.tue.mpg.de/register.php

## Create workspace and clone this repository
```
mkdir workspace
cd workspace
git clone git@github.com:boreshkinai/hybrik-transformer.git
```

## Build docker image and launch container
```
cd hybrik-transformer
docker build -f Dockerfile -t hybrik_transformer:$USER .

nvidia-docker run -p 8888:8888 -p 6000-6010:6000-6010 -v ~/workspace/hybrik-transformer:/workspace/hybrik-transformer -t -d --shm-size="16g" --name hybrik_transformer_$USER hybrik_transformer:$USER
```
Enter docker container and download data locally
```
docker exec -i -t hybrik_transformer_$USER  /bin/bash 
bash scripts/download_data_gcp.sh
```
Launch training session
```
docker exec -i -t hybrik_transformer_$USER  /bin/bash 
cd HybrIK
./scripts/train_smpl.sh hybrik_transformer ./configs/hybrik_transformer_smpl24.yaml
```

## MODEL ZOO

| Backbone | Training Data |     PA-MPJPE (3DPW)     | MPJPE (3DPW) | PA-MPJPE (Human3.6M) | MPJPE (Human3.6M) |  Download | Config |  
|----------|----------|------------|------------|-------|-----------|--------|--------------|
| ResNet-34           | w/o 3DPW | | | | | [model]() | [cfg]()    |
| ResNet-34          | w/ 3DPW | 46.0 | 74.9 | 34.6 | 50.2 | [model](https://storage.googleapis.com/hybrik-transformer/trained_models/average_model_resnet34_181_199.pth) | [cfg](./HybrIK/configs/hybrik_transformer_smpl24_w_pw3d.yaml)    |
| HRNet-W48           | w/o 3DPW | 43.4 | 73.6 | 29.8 | 48.8 | [model](https://storage.googleapis.com/hybrik-transformer/trained_models/average_model_hrnet_181_199.pth) | [cfg](./HybrIK/configs/hybrik_transformer_smpl24_hrnet.yaml)    |
| HRNet-W48          | w/ 3DPW | 42.3 | 71.6 | 29.5 | 47.5 | [model](https://storage.googleapis.com/hybrik-transformer/trained_models/average_model_hrnet_w_pw3d_181_199.pth) | [cfg](./HybrIK/configs/hybrik_transformer_smpl24_hrnet_w_pw3d.yaml)    |

