import os
import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(1, project_root)

import argparse
from detectron2.structures.masks import BitMasks
import joblib
import numpy as np
import torch
from types import SimpleNamespace

from config.defaults import DEFAULT_IMAGE_SIZE, DEFAULT_PARE_CONFIG_PATH, DEFAULT_PARE_CHECKPOINT_PATH, SMPL_FACES_PATH
from utils import local_to_global_cam


def construct_pare_args(default_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=DEFAULT_PARE_CONFIG_PATH,
                        help='config file that defines model hyperparams')

    parser.add_argument('--ckpt', type=str, default=DEFAULT_PARE_CHECKPOINT_PATH,
                        help='checkpoint path')

    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--tracker_batch_size', type=int, default=1,  # previous default: 12
                        help='batch size of object detector used for bbox tracking')

    parser.add_argument('--batch_size', type=int, default=1,  # previous default: 16
                        help='batch size of PARE')

    parser.add_argument('--smooth', action='store_true',
                        help='smooth the results to prevent jitter')

    parser.add_argument('--min_cutoff', type=float, default=0.004,
                        help='one euro filter min cutoff. '
                             'Decreasing the minimum cutoff frequency decreases slow speed jitter')

    parser.add_argument('--beta', type=float, default=1.0,
                        help='one euro filter beta. '
                             'Increasing the speed coefficient(beta) decreases speed lag.')

    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')

    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')

    parser.add_argument('--no_save', action='store_true',
                        help='disable final save of output results.')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')

    parser.add_argument('--draw_keypoints', action='store_true',
                        help='draw 2d keypoints on rendered image.')

    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    args = parser.parse_args(default_args)
    return args


def construct_default_pare_args():
    args = construct_pare_args([])
    default_values = {key: value for key, value in vars(args).items()}
    return SimpleNamespace(**default_values)


def get_pare_predictor():
    args = construct_default_pare_args()
    human_predictor = PARETester(args)
    return human_predictor


def construct_person_parameters(output_path, img_name, r, orig_h, per_bboxes, image_size=DEFAULT_IMAGE_SIZE, masks=None):
    print(output_path)
    if output_path.endswith(".pkl"):
        d = joblib.load(output_path)
        faces = np.load(SMPL_FACES_PATH)
        faces = np.expand_dims(faces.astype(np.int32), 0)

        bboxes = per_bboxes
        pare_bboxes = d['bboxes']
        inds = np.argsort(pare_bboxes[:, 0])  [:1]
        local_cam = d['pred_cam'][inds]

        verts = d['smpl_vertices']
        global_cams = local_to_global_cam(bboxes, local_cam, image_size)

        person_parameters = {
            "bboxes": per_bboxes, # bboxes[inds].astype(np.float32),
            "cams": global_cams[inds].astype(np.float32),
            "faces": faces,
            "local_cams": local_cam[inds].astype(np.float32),
            "verts": verts[inds].astype(np.float32),
            # "masks": d['pred_segm_mask'][inds].astype(np.bool)
        }

        for k, v in person_parameters.items():
            person_parameters[k] = torch.from_numpy(v).cuda()

        if masks is not None:
            full_boxes = torch.tensor([[0, 0, masks.shape[-1], masks.shape[-2]]] * len(bboxes))
            full_boxes = full_boxes.float().cuda()
            masks = BitMasks(masks).crop_and_resize(boxes=full_boxes, mask_size=image_size)
            person_parameters["masks"] = masks[inds].cuda()

        return person_parameters

    if img_name.endswith('.jpg'):
        pkl_file_name = img_name.replace(".jpg", ".pkl")
    elif img_name.endswith('.jpeg'):
        pkl_file_name = img_name.replace(".jpeg", ".pkl")
    elif img_name.endswith('.png'):
        pkl_file_name = img_name.replace("png", "pkl")

    pkl_file_name = pkl_file_name[pkl_file_name.find("/")+1:]
    pkl_file_path = os.path.join(output_path, pkl_file_name)
    d = joblib.load(pkl_file_path) # smpl

    faces = np.load(SMPL_FACES_PATH)
    faces = np.expand_dims(faces.astype(np.int32), 0)

    bboxes = per_bboxes
    pare_bboxes = d['bboxes']
    inds = np.argsort(pare_bboxes[:, 0])
    local_cam = d['pred_cam'][inds]

    verts = d['smpl_vertices']
    global_cams = local_to_global_cam(bboxes, local_cam, image_size)

    person_parameters = {
        "bboxes": per_bboxes,
        "cams": global_cams[inds].astype(np.float32),
        "faces": faces,
        "local_cams": local_cam[inds].astype(np.float32),
        "verts": verts[inds].astype(np.float32),
    }

    for k, v in person_parameters.items():
        person_parameters[k] = torch.from_numpy(v).cuda()

    if masks is not None:
        full_boxes = torch.tensor([[0, 0, image_size, image_size]] * len(bboxes))
        full_boxes = full_boxes.float().cuda()
        masks = BitMasks(masks).crop_and_resize(boxes=full_boxes, mask_size=image_size)
        person_parameters["masks"] = masks[inds].cuda()

    return person_parameters
