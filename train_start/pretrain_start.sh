# DANCE_NAME=dance16
DANCE_NAME=$1
ceph=private_alyssatan
# /apdcephfs/${ceph}/SYT/${DANCE_NAME}/
# --load_pretrain_TransG /apdcephfs/share_1364276/alyssatan/checkpoints/${DANCE_NAME}_static_train  \

# _static_train_PosePositionEmbedding

python3 ./train.py  \
--name ${DANCE_NAME}_18Feature_Temporal4_newLR_noDecay \
--batchSize 2  \
--gpu_ids 0  \
--use_laplace  \
--checkpoints_dir ../104mnt/DanceDataset/checkpoints  	  \
--pose_path ../104mnt/DanceDataset/${DANCE_NAME}/openpose_json  	 	\
--mask_path ../104mnt/DanceDataset/${DANCE_NAME}/mask 	  	\
--img_path ../104mnt/DanceDataset/${DANCE_NAME}/${DANCE_NAME} 	  	\
--densepose_path ../104mnt/DanceDataset/${DANCE_NAME}/densepose 	  	\
--bg_path ../104mnt/DanceDataset/${DANCE_NAME}/bg.jpg 	  	\
--texture_path ../104mnt/DanceDataset/${DANCE_NAME}/texture.jpg 	  	\
--flow_path ../104mnt/DanceDataset/${DANCE_NAME}/flow    \
--flow_inv_path ../104mnt/DanceDataset/${DANCE_NAME}/flow_inv \
--no_flip  \
--instance_feat  \
--input_nc 3  \
--loadSize 512  \
--resize_or_crop resize  \
--tf_log  \
--load_pretrain_TransG ../104mnt/DanceDataset/uvGenerator_pretrain_new/  \
--which_epoch_TransG 2   \
--lambda_L2 500  \
--lambda_UV 1000  \
--lambda_Prob 10  \
--use_densepose_loss  \
--save_epoch_freq 5  \
--data_ratio 0.9 \
--lambda_Temp 500 \

# --max_dataset_size 500 \
# --display_freq 10 \
# --print_freq 10 \

# --continue_train \
# --display_freq 10 \
# --print_freq 10 \

