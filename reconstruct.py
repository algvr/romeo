import os
import sys
from config.defaults import NUM_THREADS

os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import mesh_to_sdf

import cv2
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes
import glob
import joblib
import json
import neural_renderer as nr
import numpy as np
import pickle
from psbody.mesh import Mesh
from PIL import Image, ImageDraw, ImageFont
import shutil
import torch

from utils.bbox import bbox_wh_to_xy, bbox_xy_to_wh, make_bbox_square
from config.defaults import (BBOX_EXPANSION_FACTOR,
                             COCO_REMAPPINGS,
                             CONTACT_PATH, 
                             DEFAULT_CROP_INFO_DIR,
                             DEFAULT_JOINT_LR,
                             DEFAULT_LOSS_WEIGHTS,
                             DEFAULT_MASK_DIR,
                             DEFAULT_NUM_INIT_ITERS,
                             DEFAULT_NUM_JOINT_ITERS,
                             DEFAULT_OUTPUT_DIR,
                             DEFAULT_PARE_SAVE_DIR, 
                             DEFAULT_OBJECT_DEPTH_DIR,
                             DEFAULT_PERSON_DEPTH_DIR,
                             DEFAULT_USE_ADAPTIVE_ITERS,
                             DEFAULT_USE_ADAPTIVE_ITERS_INIT,
                             MAX_BBOX_CENTER_MERGING_DIST,
                             MESH_DIR,
                             MESH_MAP,
                             MINIMAL_MESH_DIR, 
                             INTERCAP_DATASET_DIR,
                             INTERCAP_KEYFRAME_PATH,
                             PATH_CLASS_NAME_DICT, 
                             PERSON_LABEL_PATH,
                             PROJECT_ROOT_DIR,
                             REND_SIZE,
                             SMPL_FACES_PATH)
from optim.init_estimation import estimate_init_object_pose, load_init_human_object_parameters, save_init_human_object_parameters, add_extra_person_parameters
from optim.joint_optimization import jointly_optimize_human_object 
from utils.geometry import center_vertices
from utils.image import load_image, segment_image
from utils.person import construct_person_parameters
from utils.phosa import PHOSA
from utils.phosa_conversions import *
from utils.zip import read_pkl_from_zip


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of dataset to use.")
    parser.add_argument("--input_dir", type=str, help="Path to directory with preprocessed images to process.")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to directory into which to store results.")
    parser.add_argument("--mask_dir", type=str, default=None, help="Path to directory with mask files for images to process.")
    parser.add_argument("--num_init_iters", type=int, default=DEFAULT_NUM_INIT_ITERS, help="Number of optimization iterations during initial estimation step.")
    parser.add_argument("--num_joint_iters", type=int, default=DEFAULT_NUM_JOINT_ITERS, help="Number of optimization iterations during joint optimization step.")
    parser.add_argument("--use_adaptive_iters_init", type=int, default=int(DEFAULT_USE_ADAPTIVE_ITERS_INIT), help="Whether to use an adaptive number of optimization iterations during the initial estimation stage.")
    parser.add_argument("--use_adaptive_iters", type=int, default=int(DEFAULT_USE_ADAPTIVE_ITERS), help="Whether to use an adaptive number of optimization iterations during the joint optimization stage.")
    parser.add_argument("--camera_idx", type=int, default=-1, help="Set to a value >= 0 to use only frames with that specific camera index in the dataset.")
    parser.add_argument("--init_only", type=int, default=0, help="Whether to run only the initial estimation step.")
    parser.add_argument("--use_pointrend", type=int, default=0, help="Whether to use PointRend for instance segmentation.")
    parser.add_argument("--visualize_masks", type=int, default=0, help="Whether to visualize segmentation masks.")
    parser.add_argument("--initial_obj_mesh", type=str, default="minimal", help='The mesh type or path to use during the initial estimation stage ("default", "minimal", or path to OBJ).')
    parser.add_argument("--joint_obj_mesh", type=str, default="minimal", help='The mesh type or path to use during the joint optimization stage ("default", "minimal", or path to OBJ).')
    parser.add_argument("--lw_penetration", type=float, default=0.0, help="Weight for penetration loss.")
    parser.add_argument("--lw_sil", type=float, default=5e4, help="Weight for occlusion-aware silhouette loss.")
    parser.add_argument("--lw_init_reldepth", type=float, default=0., help="Weight for relative depth loss during initial estimation stage.")
    parser.add_argument("--lw_reldepth", type=float, default=1e5, help="Weight for relative depth loss during joint optimization stage.")
    parser.add_argument("--lw_human_obj_contact", type=float, default=0.0, help="Weight for human-object contact loss.")
    parser.add_argument("--contact_loss_type", type=str, choices=["off", "human", "human_object"], default="off", help="Which type of contact loss to use during optimization.")
    parser.add_argument("--joint_optimization_lr", type=float, default=DEFAULT_JOINT_LR, help="Learning rate to use during the joint optimization stage.")
    parser.add_argument("--overwrite", type=int, default=0, help="Whether to overwrite results if output file already exists.")
    parser.add_argument("--crop_info_dir", type=str, default=None, help="Path to directory with cropping information for the images to process.")
    parser.add_argument("--person_est_dir", type=str, default=None, help="Path to directory with PARE estimates for the person.")
    parser.add_argument("--person_depth_dir", type=str, default=None, help="Path to directory with depth images for the person.")
    parser.add_argument("--object_depth_dir", type=str, default=None, help="Path to directory with depth images for the object.")
    parser.add_argument("--inverse_depth", type=int, default=0, help="Whether the depth maps use inverse depth.")

    args, *_ = parser.parse_known_args()

    torch.set_num_threads(NUM_THREADS)
    
    dataset = args.dataset.lower()
    contact_loss_type = args.contact_loss_type
    camera_id = args.camera_idx
    input_dir = args.input_dir
    
    if args.mask_dir in [None, ""]:
        args.mask_dir = DEFAULT_MASK_DIR.format(dataset=dataset)
    mask_dir = args.mask_dir
    
    if args.crop_info_dir in [None, ""]:
        args.crop_info_dir = DEFAULT_CROP_INFO_DIR.format(dataset=dataset)
    crop_info_dir = args.crop_info_dir
    
    if args.person_depth_dir in [None, ""]:
        args.person_depth_dir = DEFAULT_PERSON_DEPTH_DIR.format(dataset=dataset)
    person_depth_dir = args.person_depth_dir
    
    if args.object_depth_dir in [None, ""]:
        args.object_depth_dir = DEFAULT_OBJECT_DEPTH_DIR.format(dataset=dataset)
    object_depth_dir = args.object_depth_dir

    coco_class_names = np.asarray(MetadataCatalog.get("coco_2017_val").thing_classes)
    coco_person_id = np.where(coco_class_names == "person")[0][0]

    if args.lw_human_obj_contact > 0 and args.contact_loss_type == "off":
        raise ValueError("Cannot use contact_loss_type == 'off' with lw_human_obj_contact > 0")
    if args.lw_human_obj_contact == 0 and args.contact_loss_type != "off":
        raise ValueError(f"Cannot use contact_loss_type == '{args.contact_loss_type}' with lw_human_obj_contact == 0")

    contact_data = None
    if not args.init_only and contact_loss_type != "off":
        if CONTACT_PATH.lower().endswith(".json"):
            with open(CONTACT_PATH, "r") as f:
                contact_data = json.load(f)
        else:
            with open(CONTACT_PATH, "rb") as f:
                contact_data = pickle.load(f)

    if input_dir.lower().startswith("intercap_keyframes"):
        with open(INTERCAP_KEYFRAME_PATH, "r") as f:
            paths = sorted([f.format(INTERCAP_DATASET_DIR=INTERCAP_DATASET_DIR) for f in json.load(f)])
    else:
        paths = sorted([item for ext in ["jpg", "jpeg", "png", "bmp"] for item in glob.glob(os.path.join(input_dir, "**", f"*.{ext}"), recursive=True)]) 

    print(f"{paths=}")

    for path_idx, img_path in enumerate(paths):
        if not os.path.isfile(img_path):
            continue

        filename = os.path.basename(img_path)
        filename_no_ext, dot_ext = os.path.splitext(filename)

        if dot_ext.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        img_path_parts = img_path.split(os.sep)
        dataset_relative_path_parts = img_path_parts[[int(part.lower() == dataset or part.lower() == "agd20k") for part in img_path_parts].index(1)+2:]
        dataset_relative_path = os.sep.join(dataset_relative_path_parts)

        if args.lw_reldepth > 0 or args.lw_init_reldepth > 0:
            depth_path_suffix = dataset_relative_path.rsplit(".", 1)[0] + ".png"
            depth_path_person = os.path.join(person_depth_dir, depth_path_suffix)
            depth_path_object = os.path.join(object_depth_dir, depth_path_suffix)
        else:        
            depth_path_person = None 
            depth_path_object = None

        # dataset-specific exit conditions:
        if {"behave": "color" not in filename,
            "intercap": "color" not in img_path_parts,
            "agd20k": "exocentric" not in img_path_parts}.get(dataset, False):
            continue

        if camera_id >= 0:
            if {"behave": f"k{camera_id}." not in filename,
                "intercap": f"Frames_Cam{camera_id}" not in img_path}.get(dataset, False):
                continue
        
        if args.person_est_dir in ["", None]:
            args.person_est_dir = DEFAULT_PARE_SAVE_DIR.format(dataset=dataset)

        pare_res_path = None
        crop_info_path = None
        if dataset in ["behave", "intercap"]:
            pare_res_path = os.path.join(*[args.person_est_dir, *dataset_relative_path_parts[:-1], f"{filename_no_ext}.pkl"])
            crop_info_path = os.path.join(*[args.crop_info_dir, *dataset_relative_path_parts[:-1], f"{filename_no_ext}_box.json"])
        elif dataset == "agd20k":
            pare_res_path = os.path.join(args.person_est_dir, f"{filename_no_ext}.pkl")
            crop_info_path = os.path.join(*[args.crop_info_dir, f"{filename_no_ext}_box.json"])
        else:
            pare_res_path = os.path.join(args.person_est_dir, f"{filename_no_ext}.pkl")
            crop_info_path = os.path.join(*[args.crop_info_dir, f"{filename_no_ext}_box.json"])

        with open(crop_info_path, "r") as f:
            crop_info = json.load(f)

        cls_name = None
        for path_component, names in PATH_CLASS_NAME_DICT.items():
            if path_component in img_path:
                cls_name = names[0]
                break

        if cls_name is None:
            print(f"Object class could not be determined based on path (check PATH_CLASS_NAME_DICT) for {img_path}; skipping")
            continue

        if cls_name in MESH_MAP:
            obj_path = MESH_MAP[cls_name][0]
        else:
            obj_path = None
        
        if args.initial_obj_mesh:
            if args.initial_obj_mesh == "minimal":
                if dataset == "intercap":
                    obj_path = os.path.join(MINIMAL_MESH_DIR, f"{cls_name}_minimal_centered.obj")
                else:
                    obj_path = os.path.join(MINIMAL_MESH_DIR, f"{cls_name}_minimal.obj")
            elif args.initial_obj_mesh == 'default':
                if dataset == "intercap":
                    obj_path = os.path.join(MESH_DIR, f'{cls_name}_centered.obj')
                else:
                    obj_path = os.path.join(MESH_DIR, f'{cls_name}.obj')
            elif args.initial_obj_mesh.lower().endswith(".obj"):
                obj_path = args.initial_obj_mesh
            else:
                raise ValueError()

        assert os.path.exists(obj_path)

        if args.output_dir in [None, ""]:
            output_dir = DEFAULT_OUTPUT_DIR.format(dataset=dataset)
        else:
            output_dir = args.output_dir
        
        output_cls_dir = os.path.join(output_dir, *dataset_relative_path_parts[:-1])
        os.makedirs(output_cls_dir, exist_ok=True)
        print()
        print('Processing ', img_path, f'(#{path_idx+1}/{len(paths)}, class "{cls_name}")...')
        additional_name = '-'.join(({"behave": img_path_parts[-3:-1],
                                     "intercap": img_path_parts[1:-2],
                                     "agd20k": img_path_parts[1:-1]}.get(dataset, img_path_parts[:-1]))) + '-'

        cand_img, orig_w, orig_h, r = load_image(img_path, False)

        try:
            class_id = -1

            if mask_dir not in [None, ""] and not args.use_pointrend:
                person_cls = 0
                mask_data_path = os.path.join(mask_dir, os.path.dirname(dataset_relative_path), f"{filename_no_ext}.pkl.zip")
                mask_data = read_pkl_from_zip(mask_data_path)
                
                dense_mask = mask_data["csr_mask"].toarray()
                dense_mask_people = mask_data["csr_mask_people"].toarray() if "csr_mask_people" in mask_data else dense_mask
                instance_bboxes = mask_data["instance_bboxes"]
                instances = Instances(image_size=dense_mask.shape)

                obj_mask_list = []
                obj_box_list = []     
                pred_mask_list = []
                pred_box_list = []
                pred_class_list = []  # use class 0 for person, and class 1 for object 
                pred_score_list = []  # use score 1.0 for all          
                
                if "instances_to_keep_estimated" in mask_data:
                    person_mask = np.zeros_like(dense_mask).astype(int)
                    person_box = None
                    object_mask = np.zeros_like(dense_mask).astype(int)
                    object_box = None
                    for instance_id in mask_data["instances_to_keep_estimated"]:
                        instance_category = mask_data["instance_categories"][instance_id]
                        bd = instance_bboxes[instance_id]
                        instance_id = int(instance_id)
                        current_mask = np.logical_or(dense_mask == instance_id, dense_mask_people == instance_id).astype(int)
                        current_box = np.array([bd["x1"], bd["y1"], bd["x2"], bd["y2"]])

                        if instance_category == "person":
                            person_mask |= current_mask
                            if person_box is None:
                                person_box = current_box
                            else:
                                person_box = np.array([min(person_box[0], current_box[0]),
                                                        min(person_box[1], current_box[1]),
                                                        max(person_box[2], current_box[2]),
                                                        max(person_box[3], current_box[3])])
                        else:
                            object_mask |= current_mask
                            if object_box is None:
                                object_box = current_box
                            else:
                                object_box = np.array([min(object_box[0], object_box[0]),
                                                        min(object_box[1], object_box[1]),
                                                        max(object_box[2], object_box[2]),
                                                        max(object_box[3], object_box[3])])

                    if object_box is not None:
                        pred_mask_list.append(object_mask)
                        pred_box_list.append(object_box)
                        pred_class_list.append(1)
                        pred_score_list.append(1.0)
                        
                    if person_box is not None:
                        bboxes_person = np.array([person_box])
                        masks_person = torch.tensor(np.array([person_mask])).cuda()

                        pred_mask_list.append(person_mask)
                        pred_box_list.append(person_box)
                        pred_class_list.append(0)
                        pred_score_list.append(1.0)
                else:
                    print("WARNING: 'instances_to_keep_estimated' not found, using fallback instance selection strategy (SUBOPTIMAL; use depth-based selection if possible)!")

                    # merge all person boxes, keep only one
                    person_mask_list_pre = []
                    person_box_list_pre = []

                    obj_mask_list_pre = []
                    obj_box_list_pre = []

                    for instance_id, instance_category in mask_data["instance_categories"].items():
                        current_mask = np.logical_or(dense_mask == instance_id, dense_mask_people == instance_id).astype(int)
                        bd = instance_bboxes[instance_id]
                        current_box = np.array([bd["x1"], bd["y1"], bd["x2"], bd["y2"]])

                        if instance_category in ["person", "human", "man", "woman"]:
                            person_mask_list_pre.append(current_mask)
                            person_box_list_pre.append(current_box)
                        else:
                            obj_mask_list_pre.append(current_mask)
                            obj_box_list_pre.append(current_box)
                    
                    for instance_type, instance_box_list, instance_mask_list in [("person", person_box_list_pre, person_mask_list_pre), ("object", obj_box_list_pre, obj_mask_list_pre)]:

                        # union-find
                        links = {i: i for i in range(len(instance_box_list))}
                        def find(a):
                            while links[a] != a:
                                a = links[a]
                            return a

                        def link(a, b):
                            a = find(a)
                            b = find(b)
                            if a != b:
                                links[a] = b

                        for instance_a_idx in range(len(instance_box_list)):
                            instance_a_box = instance_box_list[instance_a_idx]

                            instance_a_center_x = (instance_a_box[0] + instance_a_box[2]) / 2
                            instance_a_center_y = (instance_a_box[1] + instance_a_box[3]) / 2

                            for instance_b_idx in range(instance_a_idx + 1, len(instance_box_list)):
                                instance_b_box = instance_box_list[instance_b_idx]
                                
                                instance_b_center_x = (instance_b_box[0] + instance_b_box[2]) / 2
                                instance_b_center_y = (instance_b_box[1] + instance_b_box[3]) / 2

                                center_dist = np.linalg.norm(np.array([instance_a_center_x - instance_b_center_x, instance_a_center_y - instance_b_center_y]))

                                # check if intersect or center distance small
                                x_a = max(instance_a_box[0], instance_b_box[0])
                                y_a = max(instance_a_box[1], instance_b_box[1])
                                x_b = min(instance_a_box[2], instance_b_box[2])
                                y_b = min(instance_a_box[3], instance_b_box[3])
                                intersection_area = abs(max((x_b - x_a, 0)) * max((y_b - y_a), 0))
                                if intersection_area > 0.0 or center_dist < MAX_BBOX_CENTER_MERGING_DIST:
                                    link(instance_a_idx, instance_b_idx)

                        instance_mask_dict = {}
                        instance_box_dict = {}
                        for k in links.keys():
                            vf = find(k)
                            if vf not in instance_mask_dict:
                                instance_mask_dict[vf] = instance_mask_list[vf]
                                instance_box_dict[vf] = instance_box_list[vf]

                            if vf == k:
                                continue

                            # add to vf
                            instance_mask_dict[vf] |= instance_mask_list[k]
                            instance_box_dict[vf] = np.array([min(instance_box_dict[vf][0], instance_box_list[k][0]),
                                                                min(instance_box_dict[vf][1], instance_box_list[k][1]),
                                                                max(instance_box_dict[vf][2], instance_box_list[k][2]),
                                                                max(instance_box_dict[vf][3], instance_box_list[k][3])])

                        max_instance_box_idx = -1
                        max_instance_box_area = 0.0
                        for instance_idx, instance_mask in instance_mask_dict.items():
                            instance_box_area = instance_mask.sum()
                            if instance_box_area > max_instance_box_area:
                                max_instance_box_area = instance_box_area
                                max_instance_box_idx = instance_idx

                        if max_instance_box_idx > -1:
                            pred_mask_list.append(instance_mask_dict[max_instance_box_idx])
                            pred_box_list.append(instance_box_dict[max_instance_box_idx])
                            pred_class_list.append({"person": 0, "object": 1}[instance_type])
                            pred_score_list.append(1.0)
                            
                            if instance_type == "person":
                                bboxes_person = np.array([instance_box_dict[max_instance_box_idx]])
                                masks_person = torch.tensor(np.array([instance_mask_dict[max_instance_box_idx]])).cuda()
                        else:
                            if instance_type == "person":
                                bboxes_person = np.array([])
                                masks_person = np.array([])

                instances.pred_masks = torch.tensor(pred_mask_list).cuda()
                instances.pred_boxes = Boxes(torch.tensor(pred_box_list))
                instances.pred_scores = torch.tensor(pred_score_list)
                instances.scores = torch.tensor(pred_score_list)
                instances.pred_classes = torch.tensor(pred_class_list)

                class_id = 1
            else:
                class_id = np.where(coco_class_names == COCO_REMAPPINGS.get(cls_name, cls_name))[0][0] 

                instances = segment_image(image, segmenter, False, class_id, coco_person_id, False, img_path, 0.0, False, inpainted_image=cand_img)
                if instances is None:
                    print(f"Detection failed or small IoU for {img_path}; skipping")
                    continue
                                        
                is_person = instances.pred_classes == 0
                bboxes_person = instances[is_person].pred_boxes.tensor.cpu().numpy()
                masks_person = instances[is_person].pred_masks
            
            if not os.path.exists(pare_res_path):
                print(f"No human estimation found at {pare_res_path} for {img_path}; skipping")
                continue

            if args.visualize_masks:
                mask_output_path = os.path.join(output_cls_dir, "mask_vis", img_path.replace(os.sep, "__"))
                os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)

                font = None
                font_path = os.path.join(PROJECT_ROOT_DIR, "fonts", "agane.ttf")  # Agane font by Danilo De Marco
                if os.path.isfile(font_path):
                    font = ImageFont.truetype(font_path, 30)

                with Image.open(img_path) as img_vis:
                    draw = ImageDraw.Draw(img_vis)
                    for instance_class, instance_mask in zip(instances.pred_classes, instances.pred_masks):
                        draw.bitmap((0, 0), Image.fromarray((instance_mask.cpu().numpy() * 255).astype(np.uint8)), fill=tuple(np.random.randint(0, 255, size=3)))
                    img_vis.save(mask_output_path)
                    print(f"Mask visualization saved to {mask_output_path}")
            
            depth_mask_person = np.array(Image.open(depth_path_person))
            depth_mask_object = np.array(Image.open(depth_path_object))
            
            obj_idxes = np.where(instances.pred_classes == class_id)[0]
            if len(obj_idxes) == 0:
                print(f"Object not detected for {img_path}; skipping")
                continue

            if len(obj_idxes) > 1:
                raise NotImplementedError()
            
            obj_idx = int(obj_idxes[0])
            obj_bbox = instances.pred_boxes.tensor.numpy()[obj_idx]
            person_idxes = np.where(instances.pred_classes == person_cls)[0]
            assert len(person_idxes) == 1
            person_idx = int(person_idxes[0])

            object_mask = instances.pred_masks[obj_idx].cpu().numpy()
            if object_mask.shape[0] != depth_mask_object.shape[0]:
                #assert abs(object_mask.shape[0] - depth_mask_object.shape[0] == 1)
                object_mask = cv2.resize(object_mask, (depth_mask_object.shape[0], depth_mask_object.shape[1]), interpolation=cv2.INTER_NEAREST)
            object_depth = depth_mask_object * object_mask
            
            person_mask = instances.pred_masks[person_idx].cpu().numpy()
            if person_mask.shape[0] != depth_mask_person.shape[0]:
                #assert abs(person_mask.shape[0] - depth_mask_person.shape[0] == 1)
                person_mask = cv2.resize(person_mask, (depth_mask_person.shape[0], depth_mask_person.shape[1]), interpolation=cv2.INTER_NEAREST)
            person_depth = depth_mask_person * person_mask
            
            avg_depth_obj = np.mean(object_depth[object_depth>0])
            avg_depth_person = np.mean(person_depth[person_depth>0])
            
            ratio_obj_by_person = avg_depth_obj / avg_depth_person
            
            bbox = bbox_xy_to_wh(obj_bbox)
            square_bbox = make_bbox_square(bbox, BBOX_EXPANSION_FACTOR)
            square_bbox = bbox_wh_to_xy(square_bbox)
            square_bbox = square_bbox.astype(int)
            
            left_pad = max(0, -square_bbox[0])
            top_pad = max(0, -square_bbox[1])
            right_pad = max(0, square_bbox[2] - depth_mask_object.shape[1])
            bottom_pad = max(0, square_bbox[3] - depth_mask_object.shape[0])
            square_bbox[0] += left_pad
            square_bbox[1] += top_pad
            square_bbox[2] += left_pad
            square_bbox[3] += top_pad
            depth_mask_object = np.pad(depth_mask_object, ((top_pad, bottom_pad), (left_pad, right_pad)))
            depth_mask_object = depth_mask_object[int(square_bbox[1]):int(square_bbox[3]),
                        int(square_bbox[0]):int(square_bbox[2])]
            depth_mask_object = cv2.resize(depth_mask_object.astype(np.float32), (REND_SIZE, REND_SIZE),
                                    interpolation=cv2.INTER_NEAREST)
            depth_mask_object = torch.from_numpy(depth_mask_object).unsqueeze(0)

            lw_init_depth = args.lw_init_reldepth

            with Image.open(img_path) as img:
                instances_cropped = Instances((img.height, img.width))
                instances_cropped.pred_masks = torch.tensor(instances.pred_masks[:, crop_info["y1"]:crop_info["y1"]+img.height, crop_info["x1"]:crop_info["x1"]+img.width]).cuda()
                instances_cropped.pred_boxes = Boxes(instances.pred_boxes.tensor - torch.tensor([[crop_info["x1"], crop_info["y1"], crop_info["x1"], crop_info["y1"]]]).repeat(instances.pred_boxes.tensor.shape[0], 1))
                instances_cropped.pred_scores = instances.pred_scores.clone()
                instances_cropped.scores = instances.scores.clone()
                instances_cropped.pred_classes = instances.pred_classes.clone()

                is_person = instances_cropped.pred_classes == 0
                # for camera, must use original:  # _cropped
                bboxes_person = instances[is_person].pred_boxes.tensor.cpu().numpy()
                masks_person = instances_cropped[is_person].pred_masks

                try:
                    person_parameters = construct_person_parameters(pare_res_path, filename, r, orig_h, per_bboxes=bboxes_person.astype(float), masks=masks_person,
                                                                    image_size=max(img.width, img.height))
                except Exception as ex:
                    print(f"Could not construct person from {pare_res_path} for {img_path}; skipping; detailed exception:", ex)
                    continue
                if person_parameters['verts'].shape[0] != 1:
                    print(f"Could not construct person from {pare_res_path} for {img_path} since person_parameters['verts'].shape[0] != 1; skipping")
                    continue


                object_parameters, obj_para_path, result = estimate_init_object_pose(obj_path, filename, instances_cropped, cls_name,
                                                                                     img, args.num_init_iters, os.path.join(output_cls_dir, "init_outputs"), 0, additional_name,
                                                                                     dataset, class_id=class_id, overwrite=args.overwrite,
                                                                                     depth_mask=depth_mask_object, inverse_depth=args.inverse_depth,
                                                                                     lw_depth=lw_init_depth, use_adaptive_iters=args.use_adaptive_iters_init)
            if hasattr(object_parameters, "mesh_name") and obj_path != object_parameters["mesh_name"]:
                obj_path = object_parameters["mesh_name"]  # to use the same mesh from initial estimation in the joint optimization.

            load_params = not args.overwrite and os.path.isfile(obj_para_path)
            if not load_params:
                if args.overwrite:
                    print(f"Moving {obj_para_path} to {obj_para_path}_old.bak...")
                    shutil.move(obj_para_path, obj_para_path + "_old.bak")
                if not os.path.exists(obj_para_path):
                    save_init_human_object_parameters(person_parameters, object_parameters, cls_name, 0, obj_path, img_path, obj_para_path) 
                    print(f"Initial estimation saved to {obj_para_path}")
                else:
                    object_parameters, cls_name, obj_path, img_path =\
                        load_init_human_object_parameters(obj_para_path, pare_res_path, cls_name, output_cls_dir, dataset=dataset, crop_info_path=crop_info_path, image_size=img.width)
                    print(f"Initial estimation loaded from {obj_para_path}")
            else:
                try:
                    object_parameters, cls_name, obj_path, img_path =\
                        load_init_human_object_parameters(obj_para_path, pare_res_path, cls_name, output_cls_dir, dataset=dataset, crop_info_path=crop_info_path, image_size=img.width)
                except KeyError as ex:
                    print(f"Failed to load human/object initial estimation at {obj_para_path} / {pare_res_path} for {img_path}; skipping\nDetailed exception:\n", ex)

            if "cams_new" not in person_parameters:
                with Image.open(img_path) as img:
                    add_extra_person_parameters(person_parameters, dataset, crop_info_path, REND_SIZE, max(instances.pred_masks.shape[-2:]), {"img_path": img_path})

            if object_parameters is None:
                print(f"Failed to load object initial estimation from {obj_para_path} for {img_path}; skipping")
                continue
            if object_parameters['translations'].shape[0] != 1:
                print(f"Incorrect object translation parameters at {obj_para_path} for {img_path} ({object_parameters['translations'].shape=}); skipping")
                continue
        except AssertionError as ex:
            print(f"Exception for {img_path}:", ex)
            continue

        if args.init_only:
            continue

        reconstruction_res_filename = additional_name + filename_no_ext + ".obj"
        if cls_name not in DEFAULT_LOSS_WEIGHTS:
            print(f'Class-specific loss weights for "{cls_name}" not found; using default loss weights')
            loss_weights = DEFAULT_LOSS_WEIGHTS["default"]
        else:
            print(f'Class-specific loss weights for "{cls_name}" found')
            loss_weights = DEFAULT_LOSS_WEIGHTS[cls_name]

        lw_sil = args.lw_sil
        lw_penetration = args.lw_penetration
        lw_reldepth = args.lw_reldepth

        loss_weights['lw_sil'] = lw_sil
        loss_weights['lw_penetration'] = lw_penetration
        loss_weights['lw_reldepth'] = lw_reldepth

        if args.lw_reldepth is not None:
            loss_weights['lw_reldepth'] = args.lw_reldepth

        if args.joint_obj_mesh:
            if args.joint_obj_mesh == "minimal":
                if dataset == "intercap":
                    obj_path = os.path.join(MINIMAL_MESH_DIR, f"{cls_name}_minimal_centered.obj")
                else:
                    obj_path = os.path.join(MINIMAL_MESH_DIR, f"{cls_name}_minimal.obj")
            elif args.joint_obj_mesh == "default":
                if dataset == "intercap":
                    obj_path = os.path.join(MESH_DIR, f"{cls_name}_minimal_centered.obj")
                else:
                    obj_path = os.path.join(MESH_DIR, f"{cls_name}_minimal.obj")
            elif args.joint_obj_mesh.lower().endswith(".obj"):
                obj_path = args.joint_obj_mesh
            else:
                raise NameError
        assert os.path.exists(obj_path)

        loss_weights["lw_scale"] = 0
        loss_weights["lw_scale_person"] = 0
        loss_weights["lw_human_obj_contact"] = args.lw_human_obj_contact
        print("Loss weights: " + ", ".join([f"{k}: {v}" for k, v in loss_weights.items()]))

        verts_object_og, faces_object = nr.load_obj(obj_path)
        verts_object_og, faces_object = center_vertices(verts_object_og, faces_object)
        faces_person = torch.IntTensor(np.load(SMPL_FACES_PATH).astype(int)).cuda()

        with open(PERSON_LABEL_PATH, "r") as f:
            labels_person = json.load(f)["labels"]

        labels_person = {k: torch.LongTensor(v).cuda() for k, v in labels_person.items()}
        part_fname = None 

        if part_fname:
            with open(part_fname, "r") as f:
                json_f = json.load(f)
                if "labels" in json_f.keys():
                    labels_object = json_f["labels"]
                else:
                    labels_object = json_f
            # Sometimes list is empty.
            labels_object = {k: [x - 1 for x in v] for k, v in labels_object.items() if v}
        else:
            # If no json is defined, we just map all vertices to a body part called "all".
            labels_object = {"all": np.arange(len(verts_object_og))}

        labels_object = {k: torch.LongTensor(v).cuda() for k, v in labels_object.items()}

        m = Mesh()
        m.load_from_file(obj_path)
        INT_SCALE = np.abs(m.v).max()

        if dataset == "agd20k":
            obj_parts = {}
        else:
            obj_part_file_path = os.path.join(MESH_DIR, f"{cls_name.replace('_minimal', '')}_labels.json")
            with open(obj_part_file_path, "r") as f:
                obj_parts = list(json.load(f).keys())

        if contact_loss_type != "off":
            current_contact = {}
            img_path = img_path[img_path.lower().rindex(dataset + os.sep):]
            if img_path in contact_data:
                contact_data_key = img_path
            elif cls_name in contact_data:
                contact_data_key = cls_name
            else:
                print(f"WARNING: no contact data found for {img_path}" + "; using {}")
                contact_data_key = None
                current_contact = {}
            
            if contact_data_key is not None:
                for body_part in contact_data[contact_data_key].keys():
                    # if "foot" in body_part:
                    #    continue
                    for obj_part in (["all"] if contact_loss_type == "human" else obj_parts):
                        if obj_part not in current_contact.keys():
                            current_contact[obj_part] = []
                        current_contact[obj_part].append(body_part)
                        
            # deduplication
            for obj_part in current_contact:
                current_contact[obj_part] = list(set(current_contact[obj_part]))
        else:
            current_contact = {}
        
        if contact_loss_type == "human":
            # make all object parts attract all specified human parts
            all_human_parts = []
            for obj_part in current_contact:
                all_human_parts += current_contact[obj_part]
            all_human_parts = list(set(all_human_parts))
            current_contact = {obj_part: all_human_parts for obj_part in ["all", *obj_parts]}
        
        person_mask = torch.from_numpy(cv2.resize(person_mask, (REND_SIZE, REND_SIZE), interpolation=cv2.INTER_NEAREST).astype("bool"))
        
        with Image.open(img_path) as img:
            model = PHOSA(
                translations_object=object_parameters["translations"],
                rotations_object=object_parameters["rotations"],
                verts_object_og=verts_object_og,
                faces_object=faces_object,
                target_masks=object_parameters["target_masks"],
                person_mask=person_mask,
                depth_mask=depth_mask_object,
                ratio_obj_by_person=ratio_obj_by_person,
                cams_person=person_parameters["cams_new"],
                verts_person_og=person_parameters["verts"],
                faces_person=faces_person,
                masks_object=object_parameters["masks"],
                masks_person=person_parameters["masks"],
                K_rois=object_parameters["K_roi"],
                labels_person=labels_person,
                labels_object=labels_object,
                interaction_map_parts=current_contact,
                class_name=cls_name,
                int_scale_init=INT_SCALE,
                cropped_image=img
            )

        reconstruction_res_file_dir = os.path.join(output_cls_dir, "init_outputs", "-".join([elem for elem in [
            f"init_{args.num_init_iters}", 
            f"cam_{camera_id}" if camera_id >= 0 else "",
            f"pointrend" if args.use_pointrend else "",
            "adapt" if args.use_adaptive_iters else "",
            f"init_mesh_{args.initial_obj_mesh}" if args.initial_obj_mesh in ["default", "minimal"] else "",
            *[f"{k}_{v}" for k, v in sorted([*loss_weights.items()]) if v > 0.0]] if elem not in [None, ""]]
        ))
        os.makedirs(reconstruction_res_file_dir, exist_ok=True)
        reconstruction_res_path = os.path.join(reconstruction_res_file_dir, reconstruction_res_filename)

        if not os.path.isfile(reconstruction_res_path) or args.overwrite:
            model.save_obj(reconstruction_res_path)
            print("Initial reconstruction saved to", reconstruction_res_path)
        else:
            print("Found saved initial reconstruction in", reconstruction_res_path)

        # output path
        reconstruction_res_file_dir = os.path.join(output_cls_dir, "joint_outputs", "-".join([elem for elem in [
            {"off": "", "human": "contact_human", "human_object": "contact_human_obj"}[contact_loss_type],
            f"init_{args.num_init_iters}", 
            f"jnt_{args.num_joint_iters}", 
            f"cam_{camera_id}" if camera_id >= 0 else "",
            f"pointrend" if args.use_pointrend else "",
            "adapt_" if args.use_adaptive_iters else "",
            "adapt_init" if args.use_adaptive_iters_init else "",
            f"init_mesh_{args.initial_obj_mesh}" if args.initial_obj_mesh in ["default", "minimal"] else "",
            f"joint_mesh_{args.joint_obj_mesh}" if args.joint_obj_mesh in ["default", "minimal"] else "",
            *[f"{k}_{v}" for k, v in sorted([*loss_weights.items()]) if v > 0.0]] if elem not in [None, ""]]
        ))
        os.makedirs(reconstruction_res_file_dir, exist_ok=True)

        reconstruction_res_path = os.path.join(reconstruction_res_file_dir, reconstruction_res_filename)

        try:
            if not os.path.exists(reconstruction_res_path) or args.overwrite:
                with Image.open(img_path) as img:
                    model, flag = jointly_optimize_human_object(
                                model=model,
                                class_name=cls_name,
                                loss_weights=loss_weights,
                                num_iterations=args.num_joint_iters,
                                mesh_path=obj_path,
                                contact=current_contact,
                                model_name=reconstruction_res_filename.rsplit(".", 1)[0],
                                use_adaptive_iters=args.use_adaptive_iters,
                                lr=args.joint_optimization_lr,
                                save_path=reconstruction_res_path,
                                image_path=img_path,
                                dataset=dataset
                            )
                if not flag:
                    model.save_obj(reconstruction_res_path)
                    print("Joint reconstruction saved to", reconstruction_res_path) 
                else:
                    model.save_obj(reconstruction_res_path)
                    print("No interaction found, but joint reconstruction saved anyway to", reconstruction_res_path)
            else:
                print(f"Found saved joint reconstruction in {reconstruction_res_path}; skipping")
        except ValueError as ex:
            print(f"Exception processing {reconstruction_res_path}:", ex)
            continue
