# export CUDA_VISIBLE_DEVICES="0"

python 1-train_bid.py \
  --data_name "ST" --video_dir "../data.ST_mem/videos/Train" --track_dir "../data.ST_mem/tracking/Train" --scene_classes 13 \
  --print_model 'no' \
  --pre_model "../save.ckpts/main/1-finetune_ST_0607-165004/checkpoint_1.pth.tar" \
  --snippet_inp 8 --snippet_tgt 7 --snippet_itv 12.5 \
  --lr 0.01 --batch_size 32 --iterations 32 --fr 1 \
  --epochs 50 --schedule 40 \
  --workers 16 --save_freq 2 --print_freq 50 \
  --note "" $@
