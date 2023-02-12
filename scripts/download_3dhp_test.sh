PROJECT_PATH=/workspace/hybrik-transformer
HYBRIDIK_PATH=${PROJECT_PATH}/HybrIK


mkdir -p ${HYBRIDIK_PATH}/data/3dhp && \
    cd ${HYBRIDIK_PATH}/data/3dhp && \
    wget http://gvv.mpi-inf.mpg.de/3dhp-dataset/mpi_inf_3dhp.zip && \
    unzip -o mpi_inf_3dhp.zip
    
cd ${HYBRIDIK_PATH}/data/3dhp/mpi_inf_3dhp && \
    echo "ready_to_download=1" >> conf.ig  && \
    echo "subjects=(1 2 3 4 5 6 7 8)"  >> conf.ig  && \
    echo "destination='../'"  >> conf.ig  && \
    bash get_testset.sh
    cd .. && mv mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set mpi_inf_3dhp_test_set_temp && \
    rm -r mpi_inf_3dhp_test_set && mv mpi_inf_3dhp_test_set_temp mpi_inf_3dhp_test_set
