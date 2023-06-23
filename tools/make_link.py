import os
from os import system, mkdir, link
from os.path import join, exists, realpath, abspath, dirname, basename, islink

name_list = ('save.ckpts', 'save.logs', 'save.results', 'save.tbxs', 'save.visual')
hdd_root = "/home/**/data" # NOTE: save root
proj_root = realpath('./')
proj_name = basename(proj_root)


if input(f"'{hdd_root}'\nIs this the hdd root to save the save.* dirs? (y/n)\n") != 'y':
    print("Exit!")
    exit(1)
if input(f"'{proj_root}'\nIs this the project root? (y/n)\n") != 'y':
    print("Exit!")
    exit(1)

for dir_name in name_list:
    if exists(p1:=join(hdd_root, dir_name)):
        if not exists(p2:=join(p1, proj_name)):
            mkdir(p2)
            print(f"Dir '{p2}' is created.")
            dp = join(proj_root, dir_name)
            if not exists(dp):
                system(f"ln -s {p2} {dp}")
                print(f"Link '{p2}' to '{dp}'.")
            else:
                print(f"The link {dp} has been existed!")
        else:
            print(f"'{p2}' has been existed!")
            dp = join(proj_root, dir_name)
            if not exists(dp):
                system(f"ln -s {p2} {dp}")
                print(f"Link '{p2}' to '{dp}'.")
            else:
                print(f"The link {dp} has been existed!")
    else:
        print(f"Please manually make dir: '{p1}'!")
