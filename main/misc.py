import os
from os.path import join, exists, dirname, basename, abspath, realpath
import argparse
import time
import logging
import shutil
import torch
import builtins
from tensorboardX import SummaryWriter
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple, OrderedDict as ODt


def get_proj_root() -> str:
    proj_root = dirname(dirname(realpath(__file__)))
    return proj_root


def get_time_stamp() -> str:
    _t = time.localtime()
    time_stamp = f"{str(_t.tm_mon).zfill(2)}{str(_t.tm_mday).zfill(2)}" + \
        f"-{str(_t.tm_hour).zfill(2)}{str(_t.tm_min).zfill(2)}{str(_t.tm_sec).zfill(2)}"
    return time_stamp


def get_dir_name(file=__file__) -> str:
    return basename(dirname(realpath(file)))


def get_logger(time_stamp, file_name: str = '', log_root='save.logs', data_name='') -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if file_name:
        _save_subdir = basename(dirname(abspath(__file__)))
        if data_name:
            data_name = '_' + data_name
        log_file = join(get_proj_root(), log_root, _save_subdir, f"{basename(file_name).split('.')[0]}{data_name}_{time_stamp}.log")

        if not exists(dirname(log_file)):
            os.makedirs(dirname(log_file))

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_ckpt_dir(time_stamp, file_name, ckpt_root='save.ckpts', data_name='') -> str:
    _save_subdir = basename(dirname(realpath(__file__)))
    if data_name:
        data_name = '_' + data_name
    ckpt_dir = join(get_proj_root(), ckpt_root, _save_subdir, f"{basename(file_name).split('.')[0]}{data_name}_{time_stamp}")
    if not exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    return ckpt_dir


def get_result_dir(result_root='save.results') -> str:
    result_dir = join(get_proj_root(), result_root, basename(dirname(abspath(__file__))))
    if not exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


def format_args(args: argparse.Namespace, sorted_key: bool = True) -> str:
    _cont = '\n' + '-' * 30 + "args" + '-' * 30 + '\n'
    args: dict = args.__dict__

    m_l = max([len(k) for k in args.keys()])

    key_list = list(args.keys())
    if sorted_key:
        key_list.sort()

    for _k in key_list:
        _v = args[_k]
        _cont += f"{_k:>{m_l}s} = {_v}\n"
    _cont += '-' * 60 + '\n'
    return _cont


def save_checkpoint(state, is_best, filedir, epoch, writer=builtins.print) -> None:
    if not exists(filedir):
        os.makedirs(filedir)

    filename = join(filedir, f'checkpoint_{epoch}.pth.tar')
    torch.save(state, filename)
    writer(f"Saved checkpoint to: {filename}")
    if is_best:
        shutil.copyfile(filename, join(filedir, 'checkpoint_best.pth.tar'))


class ScalarWriter():
    '''
    '''
    def __init__(self, name_list: Tuple[str], tensorboard_writer: SummaryWriter = None, init_global_step: int = 0, init_epoch: int = 0):
        self.tbxs_writer = tensorboard_writer
        self.global_step = init_global_step
        self.epoch = init_epoch
        self.epoch_value_list_dict: ODt[str, List[float]] = OrderedDict([(n, []) for n in name_list])
        self.epoch_sample_counter_list = []

    def add_step_value(self, step_value_dict: ODt[str, float], num_samples=1):
        '''
        '''
        for k, v in step_value_dict.items():
            assert k in self.epoch_value_list_dict, f"{k}"
            self.epoch_value_list_dict[k].append(v)
            if isinstance(self.tbxs_writer, SummaryWriter):
                self.tbxs_writer.add_scalar(f"step/{k}", v, self.global_step)
        self.epoch_sample_counter_list.append(num_samples)
        self.global_step += 1

    def update_epoch_average_value(self) -> ODt[str, float]:
        epoch_average_value_dict = OrderedDict()
        sc_ary: np.ndarray = np.array(self.epoch_sample_counter_list)
        for k, v in self.epoch_value_list_dict.items():
            v_ary: np.ndarray = np.array(v)
            assert v_ary.shape == sc_ary.shape
            v_mean = np.dot(v_ary, sc_ary) / np.sum(sc_ary)
            if isinstance(self.tbxs_writer, SummaryWriter):
                self.tbxs_writer.add_scalar(f"epoch/{k}", v_mean, self.epoch)
            epoch_average_value_dict[k] = v_mean

        self.epoch += 1

        for k in self.epoch_value_list_dict.keys():
            self.epoch_value_list_dict[k].clear()
        self.epoch_sample_counter_list.clear()

        return epoch_average_value_dict
