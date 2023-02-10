CONFIG=$1
CKPT=$2
PORT=${3:-23456}

HOST=$(hostname -i)

python ./validate_smpl.py \
    --batch 32 \
    --gpus 0,1,2,3 \
    --world-size 4 \
    --flip-test \
    --launcher pytorch --rank 0 \
    --dist-url tcp://${HOST}:${PORT} \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT}
