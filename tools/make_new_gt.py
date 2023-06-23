import os
from os.path import join, exists
import numpy as np

'''
convert anomaly to normality
'''

correct_vid_name_list = '''
01_0014
01_0029
01_0052
01_0138
01_0139
01_0163
'''.strip().split('\n')


src_gt_npz: dict = np.load("**/data.ST/gt.npz") # NOTE: path to the original GT file
dst_gt_dict = {}

all_test_vid_list = sorted(os.listdir("./frames/Test"))

for vid_name in all_test_vid_list:
    src_gt_ary: np.ndarray = src_gt_npz[vid_name]
    if vid_name in correct_vid_name_list:
        dst_gt_ary = np.zeros_like(src_gt_ary)
    else:
        dst_gt_ary = src_gt_ary.copy()
    dst_gt_dict[vid_name] = dst_gt_ary

np.savez("scene_gt.npz", **dst_gt_dict)
