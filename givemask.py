#copy from ui.py and remove the ui process

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np
from tools.util import *

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from open_vocab_seg import add_ovseg_config

from open_vocab_seg.utils import VisualizationDemo, VisualizationDemoIndoor

# constants
WINDOW_NAME = "Open vocabulary segmentation"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for open vocabulary segmentation")
    parser.add_argument(
        "--config-file",
        default="configs/ovseg_swinB_vitL_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        default=["/mnt/lustre/jkyang/PSG4D/sailvos3d/downloads/sailvos3d/trevor_1_int/images/000160.bmp"],
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--class-names",
        default=["person", "car", "motorcycle", "truck", "bird", "dog", "handbag", "suitcase", "bottle", "cup", "bowl", "chair", "potted plant", "bed", "dining table", "tv", "laptop", "cell phone", "bag", "bin", "box", "door", "road barrier", "stick", "lamp", "floor", "wall"],
        nargs="+",
        help="A list of user-defined class_names"
    )
    parser.add_argument(
        "--output", 
        default = "./pred",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "ovseg_swinbase_vitL14_ft_mpt.pth"],
        nargs=argparse.REMAINDER,
    )
    return parser

args = get_parser().parse_args()

def greet_sailvos3d(rgb_input, depth_map_input, class_candidates):
    args.input = rgb_input
    args.class_names = class_candidates.split(', ')
    depth_map_path = depth_map_input
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    class_names = args.class_names
    demo.run_on_image_sam(args.input, class_names, depth_map_path)
RGBroot="/mnt/ve_share/liushuai/SegmentAnyRGBD-main/sol_5_mcs_1/images"
depthroot="/mnt/ve_share/liushuai/SegmentAnyRGBD-main/sol_5_mcs_1/depth"
rgbbatch_input=sorted(os.path.join(RGBroot,i) for i in os.listdir(RGBroot))
depth_map_input=sorted(os.path.join(depthroot,i) for i in os.listdir(depthroot))
# rgbbatch_input=["/mnt/ve_share/liushuai/SegmentAnyRGBD-main/resources/demos/sailvos_1/000160.bmp"]
# depth_map_input=["/mnt/ve_share/liushuai/SegmentAnyRGBD-main/UI/sailvos3d/ex1/inputs/depth_000160.npy"]
class_candidates="person, car, motorcycle, truck, bird, dog, handbag, suitcase, bottle, cup, bowl, chair, potted plant, bed, dining table, tv, laptop, cell phone, bag, bin, box, door, road barrier, stick, lamp, floor, wall"
greet_sailvos3d(rgbbatch_input, depth_map_input, class_candidates)
