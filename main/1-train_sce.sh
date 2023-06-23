export CUDA_VISIBLE_DEVICES="3"

python 1-train_sce.py \
  --data_name "ST|Ave|Cor|NWPU" --video_dir "**" --track_dir "**" --frame_dir "**" --scene_classes N \
  --print_model 'yes' \
  --snippet_inp 1 --snippet_tgt 1 --snippet_itv 2 \
  --lr 0.01 --batch_size 32 --iterations 32 --fr 1 \
  --epochs 10 --schedule 7 \
  --workers 4 --save_freq 5 --print_freq 20 \
  --note "Train BackgroundEncoder" $@
