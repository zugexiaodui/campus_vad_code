import os
from os import listdir
from os.path import join, isdir, isfile
import shutil
import argparse

parser = argparse.ArgumentParser("")
parser.add_argument("time_stamp_list", nargs='+')
args = parser.parse_args()

save_dir_root = "../"
save_dir_name_list = ("save.ckpts", "save.logs", "save.tbxs")

for time_stamp in args.time_stamp_list:
    for save_dir_name in save_dir_name_list:
        for proj_dir_name in listdir(p1 := join(save_dir_root, save_dir_name)):
            for exp_name in listdir(p2 := join(p1, proj_dir_name)):
                exp_path = join(p2, exp_name)
                if time_stamp in exp_name:
                    if isdir(exp_path):
                        shutil.rmtree(exp_path)
                        print(f"remove the dir: \"{exp_path}\"")
                    elif isfile(exp_path):
                        os.remove(exp_path)
                        print(f"remove the file: \"{exp_path}\"")
                    else:
                        raise NotImplementedError(f"{exp_path}")
