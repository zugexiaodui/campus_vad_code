import os
from os.path import join, exists, isfile
import time
from time import time as ttime
import numpy as np
import matplotlib.pyplot as plt
import functools
import tqdm
import random
from collections import OrderedDict
from typing import Tuple

import torch
import torch.cuda
import torch.multiprocessing as mp
from torch import nn
from torch.backends import cudnn

from model_builder import SceneFrameAE, BackgroundEncoder
from dsets import TestSetTrackingObject
from misc import get_logger, get_result_dir, format_args
from metrics import cal_micro_auc
from argmanager import test_parser


def load_model(args):
    bgd_encoder = BackgroundEncoder(args.scene_classes)
    bgd_encoder.load_state_dict(torch.load(args.bgd_encoder, map_location='cpu')['model'])

    model = SceneFrameAE(inp_frm=args.snippet_inp, tgt_frm=args.snippet_tgt, bgd_encoder=bgd_encoder, lam_cvae=1.)
    if args.resume:
        if isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume, map_location='cpu')

            new_state_dict = OrderedDict()
            _prefix = 'module.'
            for k, v in checkpoint['model'].items():
                if k.startswith(_prefix):
                    new_state_dict[k[len(_prefix):]] = v
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict)

        else:
            raise FileNotFoundError("No checkpoint found at '{}'".format(args.resume))
    else:
        raise NotImplementedError("A checkpoint should be loaded.")

    model.eval()
    return model


def patch_error(img_true: torch.Tensor, img_test: torch.Tensor, patch_func_list: nn.ModuleList, lam_l1: float, use_channel_l2: bool):
    # [C,H,W]
    assert img_true.ndim == img_test.ndim == 3, f"{img_true.shape}, {img_test.shape}"
    assert img_true.shape == img_test.shape
    assert img_true.shape[0] == img_test.shape[0] == 3

    if use_channel_l2:
        diff_mse = torch.square(img_true - img_test).sum(0, True).sqrt().div(img_true.shape[0])
    else:
        diff_mse = torch.square(img_true - img_test).mean(0, True)
    diff_l1 = torch.abs(img_true - img_test).mean(0, True)

    patch_score_list = []
    for _patch_func in patch_func_list:
        _patch_err_mse: torch.Tensor = _patch_func(diff_mse)
        _patch_err_l1: torch.Tensor = _patch_func(diff_l1)
        _patch_score = _patch_err_mse.amax() + lam_l1 * _patch_err_l1.amax()
        patch_score_list.append(_patch_score)
    return patch_score_list


def frame_error(img_true: torch.Tensor, img_test: torch.Tensor, lam_l1: float, use_channel_l2: bool):
    assert img_true.ndim == img_test.ndim == 3, f"{img_true.shape}, {img_test.shape}"
    assert img_true.shape == img_test.shape
    assert img_true.shape[0] == img_test.shape[0] == 3
    if not (img_true.shape[1] > 0 and img_true.shape[2] > 0):
        print(f"\nError in `frame_error()` function: {img_true.shape}. It will be given a high value (2.0)")
        return torch.as_tensor(2.0).cpu()

    if use_channel_l2:
        diff_mse = torch.square(img_true - img_test).sum(0, True).sqrt().div(img_true.shape[0])
    else:
        diff_mse = torch.square(img_true - img_test).mean(0, True)
    diff_l1 = torch.abs(img_true - img_test).mean(0, True)

    frame_score = diff_mse.mean() + lam_l1 * diff_l1.mean()
    return frame_score.cpu()


def cal_anomaly_score(i_proc: int, proc_cnt: int, score_queue: mp.Queue, args):
    '''
    Calculate anomaly scores
    '''
    gpu_id = i_proc % torch.cuda.device_count()

    test_dataset = TestSetTrackingObject(args.video_dir, args.track_dir, args.snippet_inp + args.snippet_tgt, args.snippet_itv,
                                         device=f'cuda:{gpu_id}' if args.to_gpu else 'cpu', frame_dir=args.frame_dir)
    num_video = len(test_dataset)

    model = load_model(args)
    model.cuda(gpu_id)

    if args.print_model and i_proc == 0:
        print(model)

    fuse_func = torch.mean if args.crop_fuse_type == 'mean' else torch.amax

    if args.error_type == 'patch':
        _avg_pool_list = nn.ModuleList()
        for _patch_size in args.patch_size:
            assert 0 < _patch_size <= 256, f"{_patch_size}"
            _avg_pool_list.append(nn.AvgPool2d(_patch_size, args.patch_stride))
        snp_error_func = functools.partial(patch_error, patch_func_list=_avg_pool_list, lam_l1=args.lam_l1, use_channel_l2=args.use_channel_l2)
    elif args.error_type == 'frame':
        snp_error_func = functools.partial(frame_error, lam_l1=args.lam_l1, use_channel_l2=args.use_channel_l2)
    else:
        raise NameError(f"ERROR args.error_type: {args.error_type}")

    if not args.debug_mode:
        if not exists(args.tmp_score_dir):
            time.sleep(i_proc)
            if not exists(args.tmp_score_dir):
                os.makedirs(args.tmp_score_dir)

    if exists(args.tmp_score_dir):
        if len(os.listdir(args.tmp_score_dir)) == num_video:
            print("ATTENTION: The temp_dir is full. Check it and ensure the old dir has been emptied.")

    for vid_idx in range(i_proc, num_video, proc_cnt):
        vid_name = list(test_dataset.all_trk_dict.keys())[vid_idx]
        tmp_score_path = join(args.tmp_score_dir, f"{vid_name}.npy")
        score_dict = {}

        if exists(tmp_score_path):
            vid_scores = np.load(tmp_score_path)
        else:
            vid_stream = test_dataset[vid_idx]
            assert vid_stream.vid_name == vid_name

            if args.error_type == 'patch':
                vid_scores: np.ndarray = np.zeros([len(vid_stream), len(args.patch_size)])
            elif args.error_type == 'frame':
                vid_scores: np.ndarray = np.zeros([len(vid_stream), 1])
            else:
                raise NameError(f"ERROR args.error_type: {args.error_type}")

            tbars = functools.partial(tqdm.tqdm, desc=f"{vid_stream.vid_name}({vid_idx+1:>{len(str(num_video))}}/{num_video})", total=len(vid_stream),
                                      ncols=120, disable=False, unit='frame', position=i_proc, colour=random.choice(['green', 'blue', 'red', 'yellow', 'magenta', 'cyan', 'white']))

            for _snippet_idx in tbars(range(len(vid_stream))):
                batch_snp, background = vid_stream[_snippet_idx]
                batch_snp: torch.Tensor
                background: torch.Tensor
                if not batch_snp is None:
                    if batch_snp.device.type == 'cpu':
                        batch_snp = batch_snp.cuda(gpu_id)  # [b, c, t, h, w]
                    if background.device.type == 'cpu':
                        background = batch_snp.cuda(gpu_id)  # [b, c, h, w]

                    inp_snp = batch_snp[:, :, :args.snippet_inp]
                    tgt_snp = batch_snp[:, :, -args.snippet_tgt:]

                    with torch.no_grad():
                        inp_bgd = background.repeat(batch_snp.shape[0], 1, 1, 1)
                        out_snp, vae32, vae64 = model(inp_snp, inp_bgd)
                        out_snp: torch.Tensor

                        out_snp.squeeze_(2)
                        tgt_snp.squeeze_(2)

                        _obj_score = torch.as_tensor([snp_error_func(out_snp[i_obj], tgt_snp[i_obj]) for i_obj in range(len(batch_snp))])
                    vid_scores[_snippet_idx] = fuse_func(_obj_score, dim=0).cpu().numpy()
                else:
                    pass

            if not args.debug_mode:
                np.save(tmp_score_path, vid_scores)

        score_dict[vid_name] = vid_scores

        assert not score_queue.full()
        score_queue.put(score_dict)


if __name__ == '__main__':
    args = test_parser().parse_args()

    if not args.tmp_score_dir:
        args.tmp_score_dir = f"./{args.data_name}"
    res_stamp = f"{args.data_name}_{args.time_stamp}"

    logger = get_logger(args.time_stamp, '' if args.debug_mode else __file__, args.log_root, args.data_name)
    logger.info(format_args(args))
    if args.debug_mode:
        logger.info(f"ATTENTION: You are in DEBUG mode. Nothing will be saved!")

    cudnn.benchmark = not args.debug_mode
    torch.set_num_threads(args.threads)

    t0 = ttime()

    gt_npz: dict = np.load(args.gtnpz_path)

    if args.score_dict_path:
        logger.info(f"Using the score_dict from '{args.score_dict_path}'")
        assert exists(args.score_dict_path), f"{args.score_dict_path}"
        score_dict = np.load(args.score_dict_path)
    else:
        epoch = torch.load(args.resume, map_location='cpu')['epoch']
        args.tmp_score_dir += f"_{epoch}"

        logger.info(f"Testing epoch [{epoch}] ...")
        len_dataset = len(TestSetTrackingObject(args.video_dir, args.track_dir, args.snippet_inp + args.snippet_tgt, args.snippet_itv))
        score_queue = mp.Manager().Queue(maxsize=len_dataset)

        mp.spawn(cal_anomaly_score, args=(args.workers, score_queue, args), nprocs=args.workers)

        assert score_queue.full()
        score_dict = {}
        while not score_queue.empty():
            score_dict.update(score_queue.get())
        assert len(score_dict) == len_dataset

        # Save scores
        if not args.debug_mode:
            score_dict_path = join(get_result_dir(args.result_root), f"{res_stamp}_score_dict_{epoch}.npz")
            np.savez(score_dict_path, **score_dict)
            logger.info(f"Saved score_dict to: {score_dict_path}")
            [os.remove(join(args.tmp_score_dir, f)) for f in os.listdir(args.tmp_score_dir)]
            os.removedirs(args.tmp_score_dir)
            logger.info(f"The tmp_score_dir {args.tmp_score_dir} is removed.")

    # Calculate AUC
    origin_score_dict = OrderedDict()
    smooth_score_dict = OrderedDict()
    vid_macro_auc_dict = OrderedDict()
    default_ps = 256
    error_level_list = args.patch_size if args.error_type == 'patch' else (default_ps,)
    for _i_patch, _patch_size in enumerate(error_level_list):
        _p_score_dict = {}
        for _vid_name, _vid_score in score_dict.items():
            _p_score = _vid_score[:, _i_patch]

            if np.any(np.isnan(_p_score)):
                _p_score[np.isnan(_p_score)] = 2.
            assert not np.any(np.isnan(_p_score))
            _p_score_dict[_vid_name] = _p_score[1:] if args.ignore_first_frame_score else _p_score[:]

        _snippet_len = args.snippet_inp + args.snippet_tgt
        micro_auc = cal_micro_auc(_p_score_dict, gt_npz, _snippet_len, args.snippet_itv, score_post_process=args.score_post_process)
        origin_score_dict[_patch_size] = _p_score_dict
        logger.info(f"Patch_size {_patch_size:>3}: Micro-AUC = {micro_auc:.2%}")

    if not args.debug_mode:
        origin_score_path = join(get_result_dir(args.visual_root), f"{res_stamp}_origin_score.pkl")
        with open(origin_score_path, 'wb') as f:
            np.savez(f, origin_score_dict)
        logger.info(f"Saved origin_score to: {origin_score_path}")
        smooth_score_path = join(get_result_dir(args.visual_root), f"{res_stamp}_smooth_score.pkl")
        with open(smooth_score_path, 'wb') as f:
            np.savez(f, smooth_score_dict)
        logger.info(f"Saved smooth_score to: {smooth_score_path}")

    if not args.debug_mode:
        score_curve_dir = join(get_result_dir(args.visual_root), f"{res_stamp}_score_curve")
        if not exists(score_curve_dir):
            os.mkdir(score_curve_dir)

        for vid_name in tqdm.tqdm(sorted(score_dict.keys()), total=len(score_dict), desc="Generating score curves"):
            dst_curve_path = join(score_curve_dir, f"{vid_name}.png")
            gt_ary = gt_npz[vid_name]
            frm_idx = np.arange(len(gt_ary))
            ori_score = origin_score_dict[default_ps][vid_name]
            smo_score = smooth_score_dict[default_ps][vid_name]
            plt.figure(figsize=(12, 7), dpi=300)
            plt.plot(frm_idx, gt_ary, 'r')
            plt.plot(frm_idx, ori_score, 'b')
            plt.plot(frm_idx, smo_score, 'g')
            plt.xticks(frm_idx[:: max(int(round(len(frm_idx) // 25, -1)), 10)])
            plt.title(f"{vid_name}  AUC={vid_macro_auc_dict[default_ps][vid_name]:.2%}")
            plt.legend(['GT', 'Ori', 'Smo'])
            plt.savefig(dst_curve_path)
            plt.close()

    t1 = ttime()
    logger.info(f"Time={(t1-t0)/60:.1f} min")
