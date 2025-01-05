import os
import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(1, project_root)

import mesh_to_sdf

import argparse
import statistics
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from config.defaults import DEFAULT_IMAGE_SIZE, REND_SIZE
from utils.evaluation import eval_align_both_models_from_sRt
from utils.geometry import rot6d_to_matrix
from utils.phosa import PHOSA


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="Path to directory with reconstructions to evaluate.")
    args = parser.parse_args()
    suffix_before = "_model_before_joint_optim.pth"
    suffix_after = "_model_after_joint_optim.pth"  

    # search recursively for _model_before_joint_optim.pth, _model_after_joint_optim.pth
    
    H_dict_before = {}
    O_dict_before = {}

    H_dict_after = {}
    O_dict_after = {}

    print("Collecting paths...")

    paths = []
    for root, dirs, files in os.walk(args.input_dir):
        for fn in files:
            if fn.endswith(suffix_before):
                reconstruction_type = "before"
            elif fn.endswith(suffix_after):
                reconstruction_type = "after"
            else:
                continue
            path = os.path.join(root, fn)
            paths.append((root, fn, path, reconstruction_type))

    print(f"Found {len(paths)} paths")
    print("Evaluating...")

    progress = tqdm(sorted(paths))
    for root, fn, path, reconstruction_type in progress:
        H_dict_target = {"before": H_dict_before, "after": H_dict_after}[reconstruction_type]
        O_dict_target = {"before": O_dict_before, "after": O_dict_after}[reconstruction_type]

        path = os.path.join(root, fn)
        data_dict = torch.load(path)

        phosa_temp_init = {
            **{k: (v.squeeze(0) if k in ["faces_object", "faces_person"] else v) for k, v in data_dict["state_dict"].items() if k in ["translations_object", "rotations_object", "verts_object_og", "faces_object", "depth_mask", "cams_person", "verts_person_og",
                                                                                                                                      "faces_person", "person_mask", "ratio_obj_by_person", "masks_object", "masks_person", "K_rois"]},
            "masks_person": data_dict["state_dict"]["masks_human"],
            "labels_person": None,
            "labels_object": None,
            "interaction_map_parts": None,
            "class_name": "temp",
            "target_masks": torch.ones(1, REND_SIZE, REND_SIZE),
            "person_mask": None
        }
        phosa_obj = PHOSA(**phosa_temp_init)
        phosa_obj.load_state_dict(data_dict["state_dict"], strict=False)
        dataset = data_dict["dataset"] 
        img_path = data_dict["image_path"]
        spl = img_path.split(os.sep)

        if dataset == "behave":
            # BEHAVE/sequences/Date03_Sub05_suitcase/t0053.000/k1.color.jpg
            seq, frame, kid = img_path.split(os.sep)[-3:]
            kid = int(kid[1])
            image_id_info = {"seq": seq, "kid": kid, "frame": frame}
            key = "-".join(spl[-3:]).rsplit(".", 1)[0]
        elif dataset == "intercap":
            # InterCap/RGBD_Images/10/07/Seg_1/Frames_Cam2/color/00150.jpg
            id_sub = spl[-6]
            id_obj = spl[-5]
            id_seg = spl[-4]
            id_recon = spl[-1].rsplit(".", 1)[0]
            image_id_info = {"id_sub": id_sub, "id_obj": id_obj, "id_seg": id_seg, "id_recon": id_recon}
            key = "-".join(spl[-6:]).rsplit(".", 1)[0]
        else:
            raise NotImplementedError(f'Evaluation not implemented for dataset "{dataset}"')

        progress.set_description(key + f"_{reconstruction_type}")

        _, body_cd, obj_cd = eval_align_both_models_from_sRt(
            phosa_obj.get_verts_person()[0].detach().cpu().numpy(),
            phosa_obj.int_scales_object.detach().cpu().numpy(),
            rot6d_to_matrix(phosa_obj.rotations_object).detach().cpu().numpy()[0],
            phosa_obj.translations_object.detach().cpu().numpy()[0],
            data_dict["class_name"],
            image_id_info,
            dataset,
            align="person",
            chore_cd=True
        )
        body_cd *= 100
        obj_cd *= 100
        assert key not in H_dict_target and key not in O_dict_target, f"{key} has unexpectedly appeared more than once. Check your path for duplicates."
        H_dict_target[key] = body_cd
        O_dict_target[key] = obj_cd

    if len(O_dict_before) >= 2:
        print()
        print(f"CD before joint optimization ({len(O_dict_before)} images):\n"
              f"H: {statistics.mean(H_dict_before.values()):.2f} ± {statistics.stdev(H_dict_before.values()):.2f}\n"
              f"O: {statistics.mean(O_dict_before.values()):.2f} ± {statistics.stdev(O_dict_before.values()):.2f}\n")
        if len(O_dict_before) >= 2:
            print()
    
    if len(O_dict_after) >= 2:
        print(f"CD after joint optimization ({len(O_dict_after)} images):\n"
        f"H: {statistics.mean(H_dict_after.values()):.2f} ± {statistics.stdev(H_dict_after.values()):.2f}\n"
        f"O: {statistics.mean(O_dict_after.values()):.2f} ± {statistics.stdev(O_dict_after.values()):.2f}\n")

    if len(O_dict_before) >= 2:
        print()
        print("O before:")
        sorted_O_dict_before = sorted(O_dict_before.items(), key=lambda item: item[1], reverse=False)
        for idx, item in enumerate(sorted_O_dict_before):
            print(f"{idx:<3} {item[0]:<53} O: {item[1]:.3f}")
    
    if len(O_dict_after) >= 2:
        print()
        print("O after:")
        sorted_O_dict_after = sorted(O_dict_after.items(), key=lambda item: item[1], reverse=False)
        for idx, item in enumerate(sorted_O_dict_after):
            print(f"{idx:<3} {item[0]:<53} O: {item[1]:.3f}")
