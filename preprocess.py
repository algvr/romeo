import os
import sys
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(1, project_root)

import argparse
import json
import math
from matplotlib.image import imread
import numpy as np
import pickle
import glob
from PIL import Image
import shutil
from scipy.sparse import csr_matrix
import tempfile
import torch
from tqdm import tqdm
from types import SimpleNamespace
import zipfile

from config.defaults import (DEFAULT_PARE_CHECKPOINT_PATH,
                             DEFAULT_PARE_CONFIG_PATH,
                             INSTANCES_MAX_IMG_HEIGHT,
                             INSTANCES_MAX_IMG_WIDTH,
                             INTERCAP_DATASET_DIR,
                             INTERCAP_KEYFRAME_PATH,
                             PARE_DIR,
                             PATH_CLASS_NAME_DICT,
                             PERSON_DEPTH_MERGING_THRESHOLD,
                             OBJECT_DEPTH_MERGING_THRESHOLD)
from utils.person import construct_default_pare_args


if PARE_DIR not in sys.path:
    sys.path.insert(1, PARE_DIR)

from lang_sam import LangSAM

from pare.core.tester import PARETester
from pare.utils.demo_utils import (
    download_youtube_clip,
    video_to_images,
    images_to_video,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset of the images to preprocess.")
    parser.add_argument("--input_dir", type=str, help="Path to directory with images to preprocess.")
    parser.add_argument("--output_dir", type=str, help="Path to directory into which to store results.", default=None)
    parser.add_argument("--instances_max_img_width", type=int, help="Maximum image width for instance segmentation step.", default=INSTANCES_MAX_IMG_WIDTH)
    parser.add_argument("--instances_max_img_height", type=int, help="Maximum image height for instance segmentation step.", default=INSTANCES_MAX_IMG_HEIGHT)
    parser.add_argument("--depth_factor", type=int, default=100, help="Multiplicative factor for estimated depth.")
    parser.add_argument("--sam_type", type=str, default="sam2.1_hiera_small", help="Segment Anything model to use for instance detection with lang-sam.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", type=int, default=0)
    args = parser.parse_args()

    if args.output_dir in [None, ""]:
        args.output_dir = os.path.abspath(os.path.join("data", "preprocessed", args.dataset))
        print(f"Output directory set to {args.output_dir}")

    dataset = args.dataset
    device = args.device

    # estimate depth

    print("Collecting paths...")

    if args.input_dir.lower().startswith("intercap_keyframes"):
        with open(INTERCAP_KEYFRAME_PATH, "r") as f:
            paths = sorted([f.format(INTERCAP_DATASET_DIR=INTERCAP_DATASET_DIR) for f in json.load(f)])
    else:
        paths = []
        for root, dirs, files in os.walk(args.input_dir):
            for fn in files:
                if fn.rsplit(".", 1)[-1].lower() not in ["jpg", "jpeg", "png", "bmp"]:
                    continue

                path = os.path.join(root, fn)
                path_parts = path.split(os.sep)
                # dataset-specific exit conditions:
                if {"behave": "color" not in fn,
                    "intercap": "color" not in path_parts,
                    "agd20k": "exocentric" not in path_parts}.get(dataset, False):
                    continue

                paths.append(path)

    print(f"Found {len(paths)} paths")

    paths = sorted(paths)
    paths_relative = []
    output_paths_depth = {}
    output_paths_cropped_depth = {}
    
    for path in paths:
        img_path_parts = path.split(os.sep)
        dataset_relative_path_parts = img_path_parts[[int(part.lower() == dataset or part.lower() == "agd20k")
                                                     for part in img_path_parts].index(1)+1:]
        dataset_relative_path = os.sep.join(dataset_relative_path_parts).rsplit(".", 1)[0]
        paths_relative.append(dataset_relative_path)

    rgb_output_dir = os.path.join(args.output_dir, "cropped_rgb")
    box_output_dir = os.path.join(args.output_dir, "crop_info")
    depth_output_dir = os.path.join(args.output_dir, "depth")
    cropped_depth_output_dir = os.path.join(args.output_dir, "cropped_depth")
    instance_output_dir = os.path.join(args.output_dir, "instances")
    person_output_dir = os.path.join(args.output_dir, "person")
    person_vis_output_dir = os.path.join(args.output_dir, "person_vis")

    print("Estimating depth (1/3)...")

    zd_model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True, map_location=device)
    zd_model = zd_model.to(device)

    progress = tqdm(enumerate(paths))
    for path_idx, path in progress:
        dataset_relative_path = paths_relative[path_idx]
        output_path = os.path.join(depth_output_dir, dataset_relative_path + ".png")
        progress.set_description(f"Depth (1/3): {output_path}")
        if not os.path.isfile(output_path) or args.overwrite:
            with Image.open(path) as img:
                depth_pil = zd_model.infer_pil(img, output_type="pil")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                depth_pil.save(output_path)
        output_paths_depth[dataset_relative_path] = output_path

    del zd_model

    print("Finding instances (2/3)...")
    
    ls_model = LangSAM(sam_type=args.sam_type)
    ls_model.sam.model = ls_model.sam.model.to(device)

    progress = tqdm(enumerate(paths))
    for path_idx, path in progress:
        dataset_relative_path = paths_relative[path_idx]
        instance_output_path = os.path.join(instance_output_dir, dataset_relative_path + ".pkl.zip")
        progress.set_description(f"Instances (2/3): {instance_output_path}")

        if not os.path.isfile(instance_output_path) or args.overwrite:
            sample_class_name = None
            sample_class_repr = None
            for path_component, (class_repr, class_name, _) in PATH_CLASS_NAME_DICT.items():
                if path_component in path:
                    sample_class_name = class_name
                    sample_class_repr = class_repr
                    break

            if sample_class_name is None:
                print(f'WARNING: no candidate class found for "{path}"! Skipping.')
                continue

            sample_class_names = ["person", sample_class_name]
            sample_class_categories = ["person", sample_class_repr]
            
            output_dict = {"original_path": path,
                           "instance_categories": {},
                           "instance_bboxes": {},
                           "csr_mask": None}
            instance_idx = 1

            instance_avg_colors = {}
            instance_median_colors = {}
            instance_max_colors = {}

            with Image.open(path) as image_pil:
                image_pil_thumb = image_pil.copy()
                image_pil_thumb.thumbnail((args.instances_max_img_width, args.instances_max_img_height))
                # adapt to match original image

                output_mask_dense_thumb = np.zeros((image_pil_thumb.height, image_pil_thumb.width), dtype=int)

                for (class_repr, class_name) in zip(sample_class_categories, sample_class_names):
                    ls_output = ls_model.predict([image_pil_thumb], [class_name])[0]
                    masks, boxes = ls_output["masks"], ls_output["boxes"]
                    if len(masks) == 0:
                        continue
                    for mask_idx in range(masks.shape[0]):
                        box = ((np.array(boxes[mask_idx]) / np.array([image_pil_thumb.width, image_pil_thumb.height, image_pil_thumb.width, image_pil_thumb.height]))
                                * np.array([image_pil.width, image_pil.height, image_pil.width, image_pil.height]) )

                        output_dict["instance_categories"][instance_idx] = class_repr
                        output_dict["instance_bboxes"][instance_idx] = {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}
                        mask = masks[mask_idx].astype(bool)
                        output_mask_dense_thumb[mask] = instance_idx

                        instance_idx += 1
                
                output_mask_dense = np.array(Image.fromarray(output_mask_dense_thumb.astype(np.uint8)).resize((image_pil.width, image_pil.height), resample=Image.NEAREST))

                person_bboxes = []
                largest_bbox_area = 0.0
                for instance_id, instance_cls in output_dict["instance_categories"].items():
                    if instance_cls.lower().strip() not in ["person", "human", "man", "woman"]:
                        continue
                    if instance_id in output_dict["instance_bboxes"]:
                        box = output_dict["instance_bboxes"][instance_id]
                        person_bboxes.append(box)
                        bbox_area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
                        if bbox_area < 0:
                            raise AssertionError("bbox_area >= 0 not satisfied")
                        if bbox_area > largest_bbox_area:
                            largest_bbox_area = bbox_area
                            largest_bbox_box = [box["x1"], box["y1"], box["x2"], box["y2"]]
                
                if len(person_bboxes) == 0:
                    largest_bbox_box = [0, 0, img.width-1, img.height-1]

                box_width = largest_bbox_box[2] - largest_bbox_box[0]
                box_height = largest_bbox_box[3] - largest_bbox_box[1]

                box_width = max(box_width, box_height)
                box_height = box_width
                crop_box = (max(0, largest_bbox_box[0] - box_width / 2),
                            max(0, largest_bbox_box[1] - box_height / 4),
                            min(image_pil.width-1, largest_bbox_box[2] + box_width / 2),
                            min(image_pil.height-1, largest_bbox_box[3] + box_height / 4))
                crop_box_width = crop_box[2] - crop_box[0]
                crop_box_height = crop_box[3] - crop_box[1]
                unif_size = min(crop_box_width, crop_box_height)
                x_start = math.ceil((crop_box[0] + crop_box[2] - unif_size) / 2)
                y_start = math.ceil((crop_box[1] + crop_box[3] - unif_size) / 2)

                crop_box_new = (x_start,
                                y_start,
                                x_start + unif_size,
                                y_start + unif_size)

                cropped_img = image_pil.crop(crop_box_new)
                if cropped_img.width != cropped_img.height:
                    unif_size = min(cropped_img.width, cropped_img.height)
                    cropped_img = cropped_img.crop((0, 0, unif_size-1, unif_size-1))
                    crop_box_new[2] = crop_box_new[0] + unif_size-1
                    crop_box_new[3] = crop_box_new[1] + unif_size-1
                    
                if cropped_img.width != cropped_img.height:
                    raise AssertionError("cropped_img.width not equal to cropped_img.height")
                
                output_path = os.path.join(rgb_output_dir, dataset_relative_path + ".jpg")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cropped_img.save(output_path, quality=98)
                box_output_path = os.path.join(box_output_dir, dataset_relative_path + "_box.json")
                os.makedirs(os.path.dirname(box_output_path), exist_ok=True)
                with open(box_output_path, "w") as f:
                    json.dump({"x1": crop_box_new[0], "y1": crop_box_new[1], "x2": crop_box_new[2], "y2": crop_box_new[3]}, f)
                
                if dataset_relative_path in output_paths_depth:
                    with Image.open(output_paths_depth[dataset_relative_path]) as img:
                        img_cropped = img.crop(crop_box_new)
                    output_path = os.path.join(cropped_depth_output_dir, dataset_relative_path + ".png")
                    output_paths_cropped_depth[dataset_relative_path] = output_path
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    img_cropped.save(output_path)

            output_dict["csr_mask"] = csr_matrix(output_mask_dense)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            output_dict["instance_avg_depths_estimated"] = {}

            instances_mean_depth = {}

            if dataset_relative_path in output_paths_depth:
                depth_img_path = output_paths_depth[dataset_relative_path]
                depth_array = imread(depth_img_path)

                max_person_depth = -np.inf
                closest_object_depth = -np.inf
                has_people = any(instance_category in ["person", "human", "man", "woman"] for instance_category in output_dict["instance_categories"].values())

                for instance_id, instance_category in output_dict["instance_categories"].items():
                    instance_depth = depth_array[np.where(np.logical_and(output_mask_dense == instance_id, depth_array > 0))] * args.depth_factor
                    nonzero_mean_depth = 0 if len(instance_depth) == 0 else instance_depth.mean()
                    output_dict[f"instance_avg_depths_estimated"][instance_id] = nonzero_mean_depth
                    if output_dict["instance_categories"][instance_id].lower() in ["person", "human", "man", "woman"]:
                        if nonzero_mean_depth > max_person_depth:
                            max_person_depth = nonzero_mean_depth
                    elif not has_people and nonzero_mean_depth > closest_object_depth:
                        closest_object_depth = nonzero_mean_depth
                
                if has_people:
                    # determine closest object
                    for instance_id, instance_category in output_dict["instance_categories"].items():
                        if not output_dict["instance_categories"][instance_id].lower() in ["person", "human", "man", "woman"]:
                            nonzero_mean_depth = output_dict[f"instance_avg_depths_estimated"][instance_id]
                            if closest_object_depth == -np.inf or abs(max_person_depth - nonzero_mean_depth) < abs(max_person_depth - closest_object_depth):
                                closest_object_depth = nonzero_mean_depth

                output_dict[f"instances_to_keep_estimated"] = []
                # choose all to keep 
                for instance_id, instance_category in output_dict["instance_categories"].items():
                    nonzero_mean_depth = output_dict[f"instance_avg_depths_estimated"][instance_id]

                    if output_dict["instance_categories"][instance_id].lower() in ["person", "human", "man", "woman"]:
                        if abs(nonzero_mean_depth - max_person_depth) <= PERSON_DEPTH_MERGING_THRESHOLD:
                            output_dict[f"instances_to_keep_estimated"].append(instance_id)
                    elif abs(nonzero_mean_depth - closest_object_depth) <= OBJECT_DEPTH_MERGING_THRESHOLD:
                        output_dict[f"instances_to_keep_estimated"].append(instance_id)

            # crop instance mask, too
            output_mask_dense = output_mask_dense[crop_box_new[1]:crop_box_new[1]+img_cropped.height, crop_box_new[0]:crop_box_new[0]+img_cropped.width]
            if output_mask_dense.shape[-2] != img_cropped.height or output_mask_dense.shape[-1] != img_cropped.width:
                raise AssertionError("Mask size mismatch")

            os.makedirs(os.path.dirname(instance_output_path), exist_ok=True)
            with zipfile.ZipFile(instance_output_path, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
                zip_file.writestr(os.path.basename(instance_output_path).replace(".zip", ""), pickle.dumps(output_dict))

    del ls_model

    print("Finding person (3/3)...")
    
    pare_config = SimpleNamespace(**{
        **vars(construct_default_pare_args()),
        "cfg": DEFAULT_PARE_CONFIG_PATH,
        "ckpt": DEFAULT_PARE_CHECKPOINT_PATH,
        "mode": "folder",
        "image_folder": None,
        }
    )

    pare = PARETester(pare_config)
    pare.model = pare.model.to(args.device)

    progress = tqdm(enumerate(paths))
    for path_idx, path in progress:
        dataset_relative_path = paths_relative[path_idx]
        output_path = os.path.join(person_output_dir, dataset_relative_path + ".pkl")
        if not os.path.isfile(output_path) or args.overwrite:
            progress.set_description(f"Person (3/3): {output_path}")

            with tempfile.TemporaryDirectory() as temp_input_dir, tempfile.TemporaryDirectory() as temp_output_dir, tempfile.TemporaryDirectory() as temp_output_dir_imgs:
                os.makedirs(temp_input_dir, exist_ok=True)
                os.makedirs(temp_output_dir, exist_ok=True)
                os.makedirs(temp_output_dir_imgs, exist_ok=True)
                temp_input_path = os.path.join(temp_input_dir, os.path.basename(path))
                shutil.copy(path, temp_input_path)

                detections = pare.run_detector(temp_input_dir)
                pare.model = pare.model.float()
                os.makedirs(os.path.join(temp_output_dir, "pare_results"), exist_ok=True)
                pare.run_on_image_folder(temp_input_dir, detections, temp_output_dir, temp_output_dir_imgs, run_smplify=False)

                output_pkl_paths = glob.glob(os.path.join(temp_output_dir, "**", "*.pkl"), recursive=True)
                if len(output_pkl_paths) != 1:
                    print(f"Error reconstructing person for {path}")
                    continue
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.move(output_pkl_paths[0], output_path)

                output_img_paths = [item for ext in ["jpg", "jpeg", "png", "bmp"]
                                        for item in glob.glob(os.path.join(temp_output_dir_imgs, "**", f"*.{ext}"), recursive=True)]
                for output_img_path in output_img_paths:
                    final_img_path = os.path.join(person_vis_output_dir, dataset_relative_path, os.path.basename(output_img_path))
                    os.makedirs(os.path.dirname(final_img_path), exist_ok=True)
                    shutil.move(output_img_path, final_img_path)
    del pare
