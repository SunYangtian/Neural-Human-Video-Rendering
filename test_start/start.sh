DANCE_NAME=dance15
pid=001
cid=12


python3 ./test.py \
    --name ${DANCE_NAME}_18Feature_Temporal  \
    --checkpoints_dir ../104mnt/DanceDataset/checkpoints \
	--pose_path ./keypoints \
	--pose_tgt_path ../104mnt/DanceDataset/${DANCE_NAME}/openpose_json \
    --use_laplace \
    --bg_path ../104mnt/DanceDataset/${DANCE_NAME}/bg.jpg \
    --texture_path ../104mnt/DanceDataset/${DANCE_NAME}/texture.jpg \
    --TexG part \
    --n_downsample_global 2 \
    --n_blocks_global 10 \
    --ngf_global 48 \
    --use_mask_texture \
    --pose_plus_laplace \
	--n_downsample_bg 2 \
	--n_blocks_bg 2 \
    --no_flip \
    --instance_feat \
    --input_nc 3 \
    --loadSize 512 \
    --resize_or_crop resize \
    --results_dir ../104mnt/DanceDataset/Result/test/src_${pid}_${cid}/tgt_${DANCE_NAME} \
    --which_epoch 30

