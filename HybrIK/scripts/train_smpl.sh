EXPID=$1
CONFIG=$2
PORT=${3:-23456}

HOST=$(hostname -i)

mkdir -p 'tensorboard/'${CONFIG}'-'${EXPID} 
    
python ./train_smpl.py \
    --nThreads 8 \
    --launcher pytorch --rank 0 \
    --dist-url tcp://${HOST}:${PORT} \
    --exp-id ${EXPID} \
    --cfg ${CONFIG} --seed 123123 > 'tensorboard/'${CONFIG}'-'${EXPID}'/log.log'  2>&1 &
    
# ./scripts/train_smpl.sh hybrik_transformer_warmup_4000_hrnet_bs56_lr0.0005_w_pw3d ./configs/hybrik_transformer_smpl24_hrnet_w_pw3d.yaml
    
# ./scripts/train_smpl.sh hybrik_transformer_warmup_4000_hrnet_bs56_lr0.0005 ./configs/hybrik_transformer_smpl24_hrnet.yaml
    
# ./scripts/train_smpl.sh hybrik_transformer_warmup_6000_batch_128_lr0.00025 ./configs/hybrik_transformer_smpl24.yaml
# ./scripts/train_smpl.sh hybrik_transformer_warmup_6000 ./configs/hybrik_transformer_smpl24.yaml
# ./scripts/train_smpl.sh hybrik_transformer_blocks6_layers3_lw512_dp0.1_inv_sqrt_lr0.0005 ./configs/hybrik_transformer_smpl24.yaml

#  ./scripts/train_smpl.sh hybrik ./configs/256x192_adam_lr1e-3-res34_smpl_24_3d_base_2x_mix.yaml

# ./scripts/train_smpl.sh hybrik_transformer_custom_blocks6_layers3_lw2048_model1024_heads16_dp0.3_inv_sqrt_lr0.0003_aug_w_pw3d ./configs/hybrik_transformer_smpl24_w_pw3d.yaml
