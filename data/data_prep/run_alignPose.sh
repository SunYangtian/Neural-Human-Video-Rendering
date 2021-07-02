python3 graph_posenorm.py \
--target_keypoints /apdcephfs/share_1364276/alyssatan/dance15/openpose_json \
--source_keypoints  /apdcephfs/private_alyssatan/SYT/ft_local/006_1_2/ \
--target_shape 1024 1024 3 \
--source_shape 1024 1024 3 \
--source_frames /apdcephfs/share_1364276/alyssatan/ft_local/iPER/iPER_1024_image/006/1 \
--results /apdcephfs/private_alyssatan/SYT/ft_local/006_1_2_test \
--target_spread 400 800 \
--source_spread 400 800 \
--calculate_scale_translation