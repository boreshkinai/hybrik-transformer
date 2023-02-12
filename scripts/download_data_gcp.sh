PROJECT_PATH=/workspace/hybrik-transformer
HYBRIDIK_PATH=${PROJECT_PATH}/HybrIK
GCP_BUCKET=gs://hybrik-transformer


gsutil cp $GCP_BUCKET/model_files/* $HYBRIDIK_PATH/model_files/

mkdir -p $HYBRIDIK_PATH/data/
gsutil -m rsync -r $GCP_BUCKET/data/Hybrik $HYBRIDIK_PATH/data/

cd ${HYBRIDIK_PATH}/data/coco && \
    unzip -n val2017.zip && unzip -n train2017.zip  && unzip -n annotations_trainval2017.zip
    
cd ${HYBRIDIK_PATH}/data/pw3d && unzip -n imageFiles.zip

cp $PROJECT_PATH/scripts/extract_frames_3dhp.py ${HYBRIDIK_PATH}/data/3dhp/

cd ${HYBRIDIK_PATH}/data/3dhp && \
    unzip -n mpi_inf_3dhp_train_set.zip && \
    python extract_frames_3dhp.py && \
    unzip -n mpi_inf_3dhp_test_set.zip  
    
cd ${HYBRIDIK_PATH}/data/h36m && cat images.tar.gz* | tar --skip-old-files -zxvpf -
