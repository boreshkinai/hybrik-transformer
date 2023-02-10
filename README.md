# pose-estimation

## Register to get SMPL model
here: https://smpl.is.tue.mpg.de/register.php
https://smplify.is.tue.mpg.de/register.php

## Create workspace and clone this repository

```mkdir workspace```

```cd workspace```

```git clone git@github.com:boreshkinai/pose-estimation.git```

```cd pose-estimation && git lfs install && git lfs pull``` 

## Build docker image and launch container

Build image and start the lightweight docker container. Note that this assumes that the data for the project will be stored in the shared folder /home/pose-estimation accessible to you and other project members. 
```
docker build -f src/docker/DockerfileNoData -t pose_estimation:$USER .

nvidia-docker run -p 8888:8888 -p 6000-6010:6000-6010 -v ~/workspace/pose-estimation:/workspace/pose-estimation -v /home/pose-estimation/data:/workspace/pose-estimation/HybrIK/data -v /home/pose-estimation/model_files:/workspace/pose-estimation/HybrIK/model_files -t -d --shm-size="8g" --name pose_estimation_$USER pose_estimation:$USER

nvidia-docker run -p 8888:8888 -p 6000-6010:6000-6010 -v ~/workspace/pose-estimation:/workspace/pose-estimation -t -d --shm-size="8g" --name pose_estimation_$USER pose_estimation:$USER
```
Enter docker container and download data locally
```
docker exec -i -t pose_estimation_$USER  /bin/bash 
bash src/scripts/download_data_gcp.sh
```
Launch training session
```
cd HybrIK
./scripts/train_smpl.sh train_res34 ./configs/256x192_adam_lr1e-3-res34_smpl_3d_base_2x_mix.yaml
```

## Data are stored in the public bucket 

https://console.cloud.google.com/storage/browser/pose-estimation-data
