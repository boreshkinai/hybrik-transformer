## Step 1. Build docker image that downloads and unpacks all data within docker image

If data are already stored locally you shiuld go to Step 3 directly. This Docker can be used to assemble all data and then copy them to a permanent progect location, such as a GCP bucket, for example. The docker is huge. It makes sense to delete it after the data preparation is done. Inside docker, data will be available in folder ```HybrIK/data```. Do not forget to copy the dot below.

```
docker build -f src/docker/Dockerfile -t pose_estimation:$USER .

nvidia-docker run -p 8888:8888 -p 6000-6010:6000-6010 -v ~/workspace/pose-estimation/src:/workspace/pose-estimation/src -t -d --shm-size="1g" --name pose_estimation_$USER pose_estimation:$USER
```

This docker can also be used to run the training HybridIK training session. This can be done in the following steps.

Enter docker container
```
docker exec -i -t pose_estimation_boris  /bin/bash 
```
Go to the HybrIK derictory and launch the training script
```
cd HybrIK
./scripts/train_smpl.sh train_res34 ./configs/256x192_adam_lr1e-3-res34_smpl_3d_base_2x_mix.yaml
```
Note that the latest HybrIK version as of 15.12.2021, commit ae1bc3cea0cc5aa98fb512eeb295c3478b0c598f, is incompatible with the pytorch versions 1.9.0 nor 1.2.0 (claimed to be default in https://github.com/Jeff-sjtu/HybrIK). You will have a stack trace, which can be fixed by replacing lines 245-250 with the following code:
```
hm_x = hm_x * torch.arange(hm_x.shape[-1]).to(hm_x)
hm_y = hm_y * torch.arange(hm_y.shape[-1]).to(hm_y)
hm_z = hm_z * torch.arange(hm_z.shape[-1]).to(hm_z)
```

## Step 2. Data are stored in the public bucket 
The data assembled within the docker image described above have been transferred to the following public bucket.
https://console.cloud.google.com/storage/browser/pose-estimation-data
It is recommended to use this data source and the lightweight docker image provided below for further experimental work.

## Step 3. Build docker image that does not have any data, instead data are mounted externally

If you did not execute Step 1, skip this and proceed to building the lightweight image. If, however, you still have a container running from the Step 1, do not forget to stop and remove it. 
```
docker stop pose_estimation_$USER
docker rm pose_estimation_$USER
```
Build image and start the lightweight docker container. Note that this assumes that the data for the project will be stored in the shared folder /home/pose-estimation accessible to you and other project members
```
docker build -f src/docker/DockerfileNoData -t pose_estimation:$USER .

nvidia-docker run -p 8888:8888 -p 6000-6010:6000-6010 -v ~/workspace/pose-estimation/src:/workspace/pose-estimation/src -v ~/workspace/pose-estimation/HybrIK:/workspace/pose-estimation/HybrIK -v /home/pose-estimation/data:/workspace/pose-estimation/HybrIK/data -v /home/pose-estimation/model_files:/workspace/pose-estimation/HybrIK/model_files -t -d --shm-size="8g" --name pose_estimation_$USER pose_estimation:$USER
```
Once the new light container is running, it might be a good idea to clean docker cache at this point, if you executed Step (this should free about 1TB of storage)
```
docker container prune 
docker image prune 
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

