from tensorboardX import SummaryWriter
from model_builder import SceneFrameAE, BackgroundEncoder
from argmanager import train_parser
from misc import get_logger, format_args, get_ckpt_dir, save_checkpoint, ScalarWriter
from dsets import TrainSetTrackingObject
from torch.backends import cudnn
import torch.utils.data
from torch.nn import functional as tf
import torch.nn as nn
import os
from functools import partial
from collections import OrderedDict

import torch
import random
import numpy as np

rand_seed = 2022
random.seed(rand_seed)
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)
np.random.seed(rand_seed)


if __name__ == '__main__':
    args = train_parser().parse_args()
    logger = get_logger(args.time_stamp, '' if args.debug_mode else __file__, args.log_root, args.data_name)
    logger.info(format_args(args))
    if args.debug_mode:
        logger.info(f"ATTENTION: You are in DEBUG mode. Nothing will be saved!")

    cudnn.benchmark = not args.debug_mode

    n_gpus = torch.cuda.device_count()

    train_dataset = TrainSetTrackingObject(args.video_dir, args.track_dir, args.snippet_inp + args.snippet_tgt, args.snippet_itv, args.iterations, frame_dir=args.frame_dir)

    smpl_weight = train_dataset.vid_samp_weight * train_dataset.iterations
    assert len(smpl_weight) == len(train_dataset)
    weighted_sampler = torch.utils.data.WeightedRandomSampler(smpl_weight, len(train_dataset), replacement=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=weighted_sampler,
                                               num_workers=args.workers, pin_memory=True, drop_last=False, prefetch_factor=1)

    bgd_encoder = BackgroundEncoder(args.scene_classes)
    bgd_encoder.load_state_dict(torch.load(args.bgd_encoder, map_location='cpu')['model'])
    frame_ae: OrderedDict = torch.load(args.frame_ae, map_location='cpu')['model']

    model = SceneFrameAE(inp_frm=args.snippet_inp, tgt_frm=args.snippet_tgt, bgd_encoder=bgd_encoder, lam_cvae=1.)

    _load_info = model.load_state_dict(frame_ae, strict=False)
    print(_load_info)

    model.train()
    model = model.cuda()

    if args.print_model:
        logger.info(f"{model}")

    special_layer = ('NONE',)
    if args.fr == 0:
        for n, p in model.named_parameters():
            for _s in special_layer:
                if _s in n:
                    p.requires_grad_(False)

        logger.info("[Learning Rate] Set requires_grad ...")
        param_list = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                param_list.append(p)
                logger.info(f"{n}: requires_grad={p.requires_grad}")
            else:
                logger.info(f"{n}: requires_grad={p.requires_grad}")

        for n, m in model.named_modules():
            for _s in special_layer:
                if _s in n:
                    m.eval()
                    logger.info(f"{n} is set to {'training()' if m.training else 'evel()'} mode.")
                else:
                    logger.info(f"{n} is set to {'training()' if m.training else 'evel()'} mode.")

    elif args.fr == 1:
        logger.info("[Learning Rate] All layers have the same `lr` ")
        param_list = []
        for n, p in model.named_parameters():
            param_list.append(p)

    elif args.fr > 0:
        raise NotImplementedError("special_layer is not processed")
        param_list = [{'params': [], 'lr':args.lr},
                      {'params': [], 'lr':args.lr * args.fr}]
        pname_list = {_p['lr']: [] for _p in param_list}

        logger.info("[Learning Rate] Set finetuning_rate ...")
        for n, p in model.named_parameters():
            if p.requires_grad:
                _group_idx = 1 if special_layer in n else 0
                param_list[_group_idx]['params'].append(p)
                pname_list[param_list[_group_idx]['lr']].append(n)
            else:
                logger.info(f"{n}: requires_grad={p.requires_grad}")

        for _lr, _pn in pname_list.items():
            logger.info(f'[Optimizer] lr={_lr}:')
            logger.info(' | '.join(_pn))

    else:
        raise ValueError(f"`args.fr({args.fr})` shoule be >=0")

    criterion_mse = nn.MSELoss().cuda()
    criterion_l1 = nn.L1Loss().cuda()

    def loss_fn_cvae(recon_x: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        MSE_L1 = criterion_mse(recon_x, x.detach()) + criterion_l1(recon_x, x.detach())
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return (MSE_L1 + KLD) / x.size(0)

    optimizer = torch.optim.Adam(param_list, args.lr)
    lr_sch = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=0.1)

    _tbxs_witer = None
    if not args.debug_mode:
        ckpt_save_func = partial(save_checkpoint, is_best=False, filedir=get_ckpt_dir(args.time_stamp, __file__, args.ckpt_root, args.data_name), writer=logger.info)
        _tbxs_witer = SummaryWriter(get_ckpt_dir(args.time_stamp, __file__, args.tbxs_root, args.data_name))

    start_epoch = 1
    if args.resume:
        if os.path.exists(args.resume) and os.path.isfile(args.resume):
            ckpt = torch.load(args.resume)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_sch.load_state_dict(ckpt['lr_sch'])
            start_epoch = ckpt['epoch'] + 1
        else:
            raise FileNotFoundError(f"{args.resume}")
    else:
        if not args.debug_mode:
            ckpt_save_func({'epoch': 0,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_sch': lr_sch.state_dict()},
                           epoch=0)

    scalar_writer = ScalarWriter(('lr', 'img_mse', 'img_l1', 'loss_v32', 'loss_v64', 'loss_tot'),
                                 _tbxs_witer, (start_epoch - 1) * len(train_loader), start_epoch)

    for epoch in range(start_epoch, args.epochs + 1):
        for step, (batch_snp, batch_bgd, batch_lbl) in enumerate(train_loader):
            batch_snp: torch.Tensor
            batch_bgd: torch.Tensor
            batch_lbl: torch.Tensor
            batch_snp = batch_snp.cuda(non_blocking=True)  # [b, c, t, h, w]
            batch_bgd = batch_bgd.cuda(non_blocking=True)
            batch_lbl = batch_lbl.cuda(non_blocking=True)

            inp_snp = batch_snp[:, :, :args.snippet_inp]
            tgt_snp = batch_snp[:, :, -args.snippet_tgt:]
            inp_bgd = batch_bgd

            out_snp, vae32, vae64 = model(inp_snp, inp_bgd)

            loss_v32 = loss_fn_cvae(vae32['rec_x32'], vae32['feat_32'], vae32['mean_32'], vae32['log_var_32'])
            loss_v64 = loss_fn_cvae(vae64['rec_x64'], vae64['feat_64'], vae64['mean_64'], vae64['log_var_64'])

            img_mse: torch.Tensor = criterion_mse(out_snp, tgt_snp)
            img_l1: torch.Tensor = criterion_l1(out_snp, tgt_snp)
            loss_tot: torch.Tensor = img_mse + args.lam_l1 * img_l1 + args.lam_vae * loss_v32 + args.lam_vae * loss_v64

            optimizer.zero_grad()
            loss_tot.backward()
            optimizer.step()

            _scalar_dict = OrderedDict(lr=lr_sch.get_last_lr()[0], img_mse=img_mse.item(), img_l1=img_l1.item(), loss_v32=loss_v32.item(), loss_v64=loss_v64.item(), loss_tot=loss_tot.item())
            scalar_writer.add_step_value(_scalar_dict, len(tgt_snp))

            if step % args.print_freq == 0:
                logger.info(f"{'(DEBUG) ' if args.debug_mode else ''}Epoch[{epoch}/{args.epochs}] step {step:>4d}/{len(train_loader)}: " +
                            " ".join([f"{k}={v:.4f}" if not k in ['lr'] else f"{k}={v}" for k, v in _scalar_dict.items()]))

        _epoch_average_dict = scalar_writer.update_epoch_average_value()
        logger.info(f"Epoch[{epoch}/{args.epochs}] [ Average  ]: " +
                    " ".join([f"{k}={v:.4f}" if not k in ['lr'] else f"{k}={v}" for k, v in _epoch_average_dict.items()]))

        _last_lr = lr_sch.get_last_lr()[0]
        lr_sch.step()
        if lr_sch.get_last_lr()[0] < _last_lr:
            logger.info(f"[Learning Rate] Decay `lr` from {_last_lr} to {lr_sch.get_last_lr()[0]}")

        if not args.debug_mode:
            if epoch % args.save_freq == 0 or epoch == args.epochs:
                ckpt_save_func({'epoch': epoch,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_sch': lr_sch.state_dict()},
                               epoch=epoch)
    if not args.debug_mode:
        if isinstance(_tbxs_witer, SummaryWriter):
            _tbxs_witer.close()
