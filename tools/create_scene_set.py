import os
from os import listdir
import os.path as osp
from os.path import join, exists

'''
Make the ShanghaiTech-sd dataset.
Then use `make_new_gt.py` to create the correct scene-dependent labels.
'''

test_str_vid_list = '''
01_0014
01_0029
01_0052
01_0138
01_0139
01_0163
06_0147
06_0150
06_0155
10_0037
10_0074
12_0142
12_0148
12_0151
12_0154
12_0173
12_0174
12_0175
'''
train_str_vid_list = '''
01_0016
01_0051
01_0063
01_0073
01_0076
01_0129
01_0131
01_0134
01_0177
06_001
06_002
06_003
06_004
06_005
06_007
06_008
06_009
06_014
10_001
10_002
10_006
10_007
10_008
10_009
10_010
10_011
12_002
12_003
12_004
12_005
12_006
12_007
12_008
12_009
12_015
'''

src_root = "**"  # NOTE: dataset root path
split = 'Test'  # NOTE: 'Train' and 'Test'

scene_vid_dict = {'Train': train_str_vid_list, 'Test': test_str_vid_list}
vid_name_list = scene_vid_dict[split].strip().split('\n')
data_dir_list = {'frames': '', 'tracking': '.pkl', 'videos': '.avi'}
dst_root = os.getcwd()

for data_dir in data_dir_list:
    src_dir = join(src_root, data_dir)
    dst_split_dir = join(dst_root, data_dir, split)
    if not exists(dst_split_dir):
        os.mkdir(dst_split_dir)
    assert exists(src_dir) and exists(dst_split_dir)

    for vid_name in vid_name_list:
        dst_file_name = f"{vid_name}{data_dir_list[data_dir]}"
        dst_file_list = [join(src_dir, src_split, dst_file_name) for src_split in ('Train', 'Test')]

        src_file_or_dir = ""
        for cur_dst_file in dst_file_list:
            if exists(cur_dst_file):
                src_file_or_dir = cur_dst_file
        assert exists(src_file_or_dir)

        dst_file_or_dir = join(dst_split_dir, dst_file_name)

        if exists(dst_file_or_dir):
            continue

        print(src_file_or_dir, dst_file_or_dir)

        if osp.isfile(src_file_or_dir):
            os.system(f"ln -s {src_file_or_dir} {dst_file_or_dir}")
            # os.system(f"cp {src_file_or_dir} {dst_file_or_dir}")
        elif osp.isdir(src_file_or_dir):
            os.system(f"ln -s {src_file_or_dir} {dst_file_or_dir}")
            # os.system(f"cp -r {src_file_or_dir} {dst_file_or_dir}")
        else:
            raise NotImplementedError(src_file_or_dir)
