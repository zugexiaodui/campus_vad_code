export CUDA_VISIBLE_DEVICES="1,3"

python 2-test_ft.py \
  --data_name "ST" --video_dir "../data.ST_mem/videos/Test" --track_dir "../data.ST_mem/tracking/Test" --gtnpz_path "../data.ST_mem/gt.npz" --frame_dir "../data.ST_mem/frames/Test" --scene_classes 13 \
  --print_model 'yes' \
  --bgd_encoder "../save.ckpts/main/1-train_sce_ST_0607-152755/checkpoint_1.pth.tar" \
  --resume "../save.ckpts/main/1-finetune_ST_0607-165004/checkpoint_1.pth.tar" \
  --snippet_inp 8 --snippet_tgt 1 --snippet_itv 2 \
  --error_type "patch" --patch_size 256 128 64 32 16 --patch_stride 8 --use_channel_l2 --lam_l1 1.0 --crop_fuse_type "max" \
  --score_post_process "filt" \
  --workers 2 --to_gpu --threads 48 --note "" $@
# for ST: --score_post_process "filt" "norm"