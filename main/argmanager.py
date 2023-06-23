import argparse
from misc import get_time_stamp


def str2bool(v: str):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expect boolean value in string.')


def _base_parser(training: bool):
    parser = argparse.ArgumentParser(description=f'')

    # Path settings
    parser.add_argument('--data_name', type=str, default='',
                        help='Name of the dataset.')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Path to the videos.')
    parser.add_argument('--track_dir', type=str, required=True,
                        help='Path to the tracks.')
    parser.add_argument('--frame_dir', type=str, default='',
                        help='Path to the frames.')

    parser.add_argument('--time_stamp', type=str, default=get_time_stamp(),
                        help='')

    parser.add_argument('--resume', default="", type=str,
                        help='')
    parser.add_argument('--bgd_encoder', default="", type=str,
                        help='')
    parser.add_argument('--frame_ae', default="", type=str,
                        help='')

    parser.add_argument('--scene_classes', type=int,
                        help='')

    # Resource usage settings
    parser.add_argument('--workers', default=6 if training else 1, type=int,
                        help='Number of data loading workers')

    # Model settings
    parser.add_argument('--lam_cvae', type=float, default=0.,
                        help="")

    # Dataset settings
    parser.add_argument('--snippet_inp', type=int, default=8,
                        help="")
    parser.add_argument('--snippet_itv', type=float, default=2,
                        help="")
    parser.add_argument('--snippet_tgt', type=int, default=1,
                        help="")

    # Loss weight settings
    parser.add_argument('--lam_l1', default=1.0, type=float,
                        help="")
    parser.add_argument('--lam_vae', default=1.0, type=float,
                        help="")

    # Other settings
    parser.add_argument('--log_root', default="save.logs", type=str,
                        help='')
    parser.add_argument('--note', default="", type=str,
                        help='A note for this experiment')
    parser.add_argument('--print_model', type=str2bool, default='yes',
                        help='')
    parser.add_argument('--debug_mode', action="store_true",
                        help='')

    return parser


def train_parser():
    parser = _base_parser(training=True)

    # Optimizer settings
    parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                        help='Learning rate.')
    parser.add_argument('--fr', '--funetune_rate', default=1, type=float,
                        help='')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', default=80, type=int,
                        help='Number of total epochs to run.')
    parser.add_argument('--schedule', default=[60], nargs='*', type=int,
                        help='Learning rate schedule.')

    # Dataset settings
    parser.add_argument('--iterations', default=32, type=int,
                        help='A way to simulate more epochs.')

    # Saving and logging settings
    parser.add_argument('--ckpt_root', default="save.ckpts", type=str,
                        help='')
    parser.add_argument('--tbxs_root', default="save.tbxs", type=str,
                        help='')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='Save frequency.')
    parser.add_argument('--print_freq', default=10, type=int,
                        help='Print frequency.')

    parser.add_argument('--pre_model', default="", type=str,
                        help='the pre-trained forward model')

    return parser


def test_parser():
    parser = _base_parser(training=False)

    # Path settings
    parser.add_argument('--gtnpz_path', type=str, required=True,
                        help='Path to groundtruth npz file.')
    parser.add_argument('--score_dict_path', type=str, default="",
                        help='Only calculate AUCs for this score_dict. --video_dir and --resume will be ignored.')

    # Dataset settings
    parser.add_argument('--to_gpu', action='store_true',
                        help="put data to gpu")
    parser.add_argument("--ignore_first_frame_score", action="store_true")

    # Resource usage settings
    parser.add_argument('--threads', default=24, type=int,
                        help='Number of threads used by pytorch')

    # Error settings
    parser.add_argument('--error_type', type=str, default='frame', choices=('frame', 'patch'),
                        help='')
    parser.add_argument('--patch_size', type=int, nargs='+',
                        help='')
    parser.add_argument('--patch_stride', type=int, default=8,
                        help='')
    parser.add_argument('--use_channel_l2', action="store_true",
                        help='')
    parser.add_argument('--crop_fuse_type', type=str, default='mean', choices=('mean', 'max'),
                        help='Use mean or max to obtaion snippet_score')

    parser.add_argument('--score_post_process', type=str, nargs='*', default=['filt'], choices=('filt', 'norm'),
                        help='')

    # Saving settings
    parser.add_argument('--tmp_score_dir', default="", type=str,
                        help='')
    parser.add_argument('--result_root', default="save.results", type=str,
                        help='')
    parser.add_argument('--visual_root', default="save.visual", type=str,
                        help='')
    return parser


if __name__ == '__main__':
    pass
