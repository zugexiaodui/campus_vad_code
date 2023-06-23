export CUDA_VISIBLE_DEVICES="3"

python 1-train_cvae.py \
  --data_name "ST" --video_dir "../data.ST_mem/videos/Train" --track_dir "../data.ST_mem/tracking/Train" --scene_classes 13 \
  --print_model 'yes' \
  --bgd_encoder "../save.ckpts/main/1-train_sce_ST_0607-152755/checkpoint_1.pth.tar" \
  --frame_ae "../save.ckpts/main/1-train_ae_ST_0607-153012/checkpoint_1.pth.tar" \
  --snippet_inp 8 --snippet_tgt 1 --snippet_itv 2 \
  --lr 0.01 --batch_size 32 --iterations 160 --lam_vae 0.1 --fr 0 \
  --epochs 16 --schedule 12 \
  --workers 16 --save_freq 4 --print_freq 50 \
  --note "Train the CVAE in SceneFrameAE" $@
