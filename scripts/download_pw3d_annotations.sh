PROJECT_PATH=/workspace/pose-estimation
HYBRIDIK_PATH=${PROJECT_PATH}/HybrIK


mkdir -p ${HYBRIDIK_PATH}/data/pw3d/json && \
    cd ${HYBRIDIK_PATH}/data/pw3d/json && \
    gdown https://drive.google.com/drive/folders/1f7DyxyvlC9z6SFT37eS6TTQiUOXVR9rK -O ${HYBRIDIK_PATH}/data/pw3d/json --folder
    