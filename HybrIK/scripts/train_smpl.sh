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
    
# nohup python ./train_smpl.py \
#             --nThreads 6 \
#             --launcher pytorch --rank 0 \
#             --dist-url tcp://${HOST}:${PORT} \
#             --exp-id ${EXPID} \
#             --cfg './configs/'${CONFIG} --seed 123123 \
#              > 'tensorboard/'${CONFIG}'-'${EXPID}'/log.log' 2>&1 &

# ./scripts/train_smpl.sh hybrik_transformer_blocks6_layers3_lw512_dp0.1_multistep150_200_lr0.000125_gamma0.5_aug ./configs/hybrik_transformer_smpl24.yaml

# ./scripts/train_smpl.sh hybrik_transformer_blocks6_layers3_lw512_dp0.1_inv_sqrt_lr0.0005_no_aug ./configs/hybrik_transformer_smpl24.yaml

#  ./scripts/train_smpl.sh hybrik ./configs/256x192_adam_lr1e-3-res34_smpl_24_3d_base_2x_mix.yaml

# ./scripts/train_smpl.sh hybrik_transformer_custom_blocks6_layers3_lw2048_model1024_heads16_dp0.3_inv_sqrt_lr0.0003_aug_w_pw3d ./configs/hybrik_transformer_smpl24_w_pw3d.yaml

# ./scripts/train_smpl.sh hybrik_transformer_blocks6_layers3_lw512_dp0.1_inv_sqrt_lr0.0005_aug_w_pw3d ./configs/hybrik_transformer_smpl24_w_pw3d.yaml

# ./scripts/train_smpl.sh hybrik_transformer_blocks6_layers3_lw512_dp0.1_inv_sqrt_lr0.0005_aug ./configs/hybrik_transformer_smpl24.yaml


# ./scripts/train_smpl.sh hybrik_transformer_blocks2_layers3_lw512_dp0.0_inv_sqrt_lr0.0005_1 ./configs/hybrik_transformer_smpl24.yaml

# ./scripts/train_smpl.sh hybrik_transformer_blocks2_layers3_lw2048_dp0.1_inv_sqrt_lr0.0005 ./configs/hybrik_transformer_smpl24.yaml


########### OLD UNITY TIME EXPERIMENTS ####################
    
# nohup python ./train_smpl.py \
#             --nThreads 8 \
#             --launcher pytorch --rank 0 \
#             --dist-url tcp://${HOST}:${PORT} \
#             --exp-id ${EXPID} \
#             --cfg './configs/'${CONFIG} --seed 123123 \
#              > 'tensorboard/'${CONFIG}'-'${EXPID}'/log.log' 2>&1 &

#  ./scripts/train_smpl.sh transformer_full_2block_multistep_lr0.0002_gamma0.1_uvd_hybrik ./configs/deeppose_transformer_smpl24.yaml

#  ./scripts/train_smpl.sh transformer_uvd_2block_inv_sqrt_lr0.0005_uvd1_smpl240.1_quat ./configs/deeppose_transformer_ne_smpl24.yaml


#  ./scripts/train_smpl.sh transformer_uvd_2block_lw512_inv_sqrt_lr0.0005_dp0.1_uvd_hybrik ./configs/deeppose_transformer_smpl24.yaml

#  ./scripts/train_smpl.sh transformer_all_uvd_2block_lw512_inv_sqrt_lr0.0005_dp0.1_uvd_hybrik ./configs/deeppose_transformer_smpl24.yaml