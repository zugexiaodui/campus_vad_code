from tensorboardX import SummaryWriter
from model_builder import BidirectionalFrameAE
from argmanager import train_parser
from misc import get_logger, format_args, get_ckpt_dir, save_checkpoint, ScalarWriter
from dsets import TrainSetTrackingObject
from torch.backends import cudnn
import torch.utils.data
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

    snippet_cur = 1
    train_dataset = TrainSetTrackingObject(args.video_dir, args.track_dir, args.snippet_inp + args.snippet_tgt, args.snippet_itv, args.iterations, frame_dir=args.frame_dir)

    smpl_weight = train_dataset.vid_samp_weight * train_dataset.iterations
    assert len(smpl_weight) == len(train_dataset)
    weighted_sampler = torch.utils.data.WeightedRandomSampler(smpl_weight, len(train_dataset), replacement=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=weighted_sampler,
                                               num_workers=args.workers, pin_memory=True, drop_last=False, prefetch_factor=1)

    model = BidirectionalFrameAE(inp_frm=args.snippet_inp, tgt_frm=args.snippet_tgt, scene_classes=args.scene_classes)
    pre_state = torch.load(args.pre_model, map_location='cpu')

    for n, p in model.named_parameters():
        if n.startswith('f_frameAE'):
            pre_w: torch.Tensor = pre_state['model'][n.replace('f_frameAE.', '')].data
            if p.data.shape != pre_w.shape:
                assert 'rec_conv.3' in n
                p.data[:3, ...] = pre_w[:, ...]
            else:
                p.data = pre_w
        else:
            pre_w: torch.Tensor = pre_state['model'][n.replace('b_frameAE.', '')].data
            assert p.data.shape == pre_w.shape, f"{n}, {p.data.shape}, {pre_w.shape}"
            p.data = pre_w

    # for n, p in model.named_parameters():
    #     if 'bn' in n:
    #         print(n, p)

    model.train()
    model = model.cuda()

    if args.print_model:
        logger.info(f"{model}")

    # special_layer = ('bgd_encoder', 'f_frameAE', 'b_frameAE')
    # special_layer = ('bgd_encoder', 'f_frameAE')
    special_layer = ('bgd_encoder',)
    # exclude_layer = 'rec_conv.3'
    exclude_layer = 'rec_conv'
    if args.fr == 0:
        for n, p in model.named_parameters():
            for _s in special_layer:
                if _s in n and not exclude_layer in n:
                    p.requires_grad_(False)

        logger.info("[Learning Rate] Set requires_grad ...")
        param_list = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                param_list.append(p)
                logger.info(f"{n}: requires_grad={p.requires_grad}")
            # else:
            #     logger.info(f"{n}: requires_grad={p.requires_grad}")

        model.train()
        for n, m in model.named_modules():
            for _s in special_layer:
                if _s in n and not exclude_layer in n:
                    m.eval()
        for n, m in model.named_modules():
            if m.training:
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

    optimizer_f = torch.optim.Adam(filter(lambda p: p.requires_grad, model.f_frameAE.parameters()), args.lr)
    optimizer_b = torch.optim.Adam(filter(lambda p: p.requires_grad, model.b_frameAE.parameters()), args.lr)
    lr_sch_f = torch.optim.lr_scheduler.MultiStepLR(optimizer_f, args.schedule, gamma=0.1)
    lr_sch_b = torch.optim.lr_scheduler.MultiStepLR(optimizer_b, args.schedule, gamma=0.1)

    _tbxs_witer = None
    if not args.debug_mode:
        ckpt_save_func = partial(save_checkpoint, is_best=False, filedir=get_ckpt_dir(args.time_stamp, __file__, args.ckpt_root, args.data_name), writer=logger.info)
        _tbxs_witer = SummaryWriter(get_ckpt_dir(args.time_stamp, __file__, args.tbxs_root, args.data_name))

    start_epoch = 1
    if args.resume:
        if os.path.exists(args.resume) and os.path.isfile(args.resume):
            ckpt = torch.load(args.resume)
            model.load_state_dict(ckpt['model'])
            optimizer_f.load_state_dict(ckpt['optimizer_f'])
            optimizer_b.load_state_dict(ckpt['optimizer_b'])
            lr_sch_f.load_state_dict(ckpt['lr_sch_f'])
            lr_sch_b.load_state_dict(ckpt['lr_sch_b'])
            start_epoch = ckpt['epoch'] + 1
        else:
            raise FileNotFoundError(f"{args.resume}")
    else:
        if not args.debug_mode:
            ckpt_save_func({'epoch': 0,
                            'model': model.state_dict(),
                            'optimizer_f': optimizer_f.state_dict(),
                            'optimizer_b': optimizer_b.state_dict(),
                            'lr_sch_f': lr_sch_f.state_dict(),
                            'lr_sch_b': lr_sch_b.state_dict()},
                           epoch=0)

    def loss_fn_cvae(recon_x: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        MSE_L1 = criterion_mse(recon_x, x.detach()) + criterion_l1(recon_x, x.detach())
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return (MSE_L1 + KLD) / x.size(0)

    scalar_writer = ScalarWriter(('lr', 'f_loss', 'b_loss'),
                                 _tbxs_witer, (start_epoch - 1) * len(train_loader), start_epoch)

    for epoch in range(start_epoch, args.epochs + 1):
        for step, (batch_snp, batch_bgd, batch_lbl) in enumerate(train_loader):
            batch_snp: torch.Tensor = batch_snp.cuda(non_blocking=True)  # [b, c, t, h, w]
            batch_bgd: torch.Tensor = batch_bgd.cuda(non_blocking=True)
            batch_lbl: torch.Tensor = batch_lbl.cuda(non_blocking=True)

            f_inp_snp = batch_snp[:, :, :args.snippet_inp]
            f_tgt_snp = batch_snp[:, :, - args.snippet_tgt:]
            f_out_snp, f_vae32, f_vae64 = model.f_frameAE(f_inp_snp, batch_bgd)  # [b, c, tgt_frm, h, w]
            # f_out_snp = model.f_frameAE(f_inp_snp, batch_bgd)  # [b, c, tgt_frm, h, w]

            f_loss: torch.Tensor = criterion_mse(f_out_snp, f_tgt_snp) + args.lam_l1 * criterion_l1(f_out_snp, f_tgt_snp)

            optimizer_f.zero_grad()
            f_loss.backward()
            optimizer_f.step()

            # Only randomly select one backward snippet for training
            _i_snp = random.randrange(0, args.snippet_tgt)

            b_all_snp_true = torch.flip(batch_snp[:, :, _i_snp: _i_snp + args.snippet_inp + snippet_cur], [2])
            b_inp_snp_true = b_all_snp_true[:, :, :args.snippet_inp]
            b_tgt_snp_true = b_all_snp_true[:, :, args.snippet_inp: args.snippet_inp + snippet_cur]

            b_all_snp_fcst = torch.flip(torch.cat([f_inp_snp, f_out_snp.detach()], 2)[:, :, _i_snp: _i_snp + args.snippet_inp + snippet_cur], [2])
            b_inp_snp_fcst = b_all_snp_fcst[:, :, :args.snippet_inp]
            b_tgt_snp_fcst = b_all_snp_fcst[:, :, args.snippet_inp: args.snippet_inp + snippet_cur]

            b_inp_snp = torch.cat([b_inp_snp_true, b_inp_snp_fcst], 0)
            b_tgt_snp = torch.cat([b_tgt_snp_true, b_tgt_snp_fcst], 0)
            b_batch_bgd = torch.cat([batch_bgd, batch_bgd], 0)
            b_out_snp, b_vae32, b_vae64 = model.b_frameAE(b_inp_snp, b_batch_bgd)
            # b_out_snp = model.b_frameAE(b_inp_snp, batch_bgd)

            b_loss: torch.Tensor = criterion_mse(b_out_snp, b_tgt_snp) + args.lam_l1 * criterion_l1(b_out_snp, b_tgt_snp)

            optimizer_b.zero_grad()
            b_loss.backward()
            optimizer_b.step()

            _scalar_dict = OrderedDict(lr=lr_sch_f.get_last_lr()[0], f_loss=f_loss.item(), b_loss=b_loss.item())
            scalar_writer.add_step_value(_scalar_dict, len(batch_snp))

            if step % args.print_freq == 0:
                logger.info(f"{'(DEBUG) ' if args.debug_mode else ''}Epoch[{epoch}/{args.epochs}] step {step:>4d}/{len(train_loader)}: " +
                            " ".join([f"{k}={v:.4f}" if not k in ['lr'] else f"{k}={v}" for k, v in _scalar_dict.items()]))

        _epoch_average_dict = scalar_writer.update_epoch_average_value()
        logger.info(f"Epoch[{epoch}/{args.epochs}] [ Average  ]: " +
                    " ".join([f"{k}={v:.4f}" if not k in ['lr'] else f"{k}={v}" for k, v in _epoch_average_dict.items()]))

        _last_lr = lr_sch_f.get_last_lr()[0]
        lr_sch_f.step()
        lr_sch_b.step()
        if lr_sch_f.get_last_lr()[0] < _last_lr:
            logger.info(f"[Learning Rate] Decay `lr` from {_last_lr} to {lr_sch_f.get_last_lr()[0]}")

        if not args.debug_mode:
            if epoch % args.save_freq == 0 or epoch == args.epochs:
                ckpt_save_func({'epoch': epoch,
                                'model': model.state_dict(),
                                'optimizer_f': optimizer_f.state_dict(),
                                'optimizer_b': optimizer_b.state_dict(),
                                'lr_sch_f': lr_sch_f.state_dict(),
                                'lr_sch_b': lr_sch_b.state_dict()},
                               epoch=epoch)
    if not args.debug_mode:
        if isinstance(_tbxs_witer, SummaryWriter):
            _tbxs_witer.close()
