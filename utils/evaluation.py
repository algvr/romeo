import os
import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(1, project_root)

import cv2
import glob
import json
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import neural_renderer as nr
import numpy as np
import open3d as o3d
import os
from psbody.mesh import Mesh
from PIL import Image
import sys
import torch
from typing import Optional

from config.defaults import DEFAULT_IMAGE_SIZE, BEHAVE_DATASET_DIR, BEHAVE_GT_ORIGIN_PATH, INTERCAP_DATASET_DIR, INTERCAP_GT_ORIGIN_PATH, PROJECT_ROOT_DIR
from utils.geometry import center_vertices
from utils.chamfer_distance_chore import chamfer_distance_chore  # the chamfer distance used by CHORE
import utils.chamfer_distance_torch as cd_torch_module
chamfer_distance_torch = cd_torch_module.chamferDist()
from utils.procrustes_alignment import compute_similarity_transform_batch


def vis_external_pointcloud(xyz, name):
    """
    Visualize a pointcloud from a remote serve on a local machine, using open3d.
    IMPORTANT: Before using this function, you must fist set up a ssh tunnel from remote server to your local machine
    PLEASE FOLLOW THE INSTRUCTIONS in https://github.com/isl-org/Open3D/issues/2789 TO USE THIS FUNCTION!

    Args:
        xyz: numpy array of shape N x 3
        name: name of the pointcloud

    Returns:
        draw the pointclod on local machine
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    import random
    a = random.random() * 100
    draw = o3d.visualization.EV.draw
    draw([{'geometry': pcd, 'name': name+f'_{a:.0f}'}])


def render_view(phosa, theta=1.57, d=2.5, view="top", ori_look_down_angle=0.1, image_size=DEFAULT_IMAGE_SIZE):
    """
    Args:
        phosa: the phosa object
        theta: rotation degree
        d: distance from camera to object
        view: render view, could be either 'top' or 'left'
        ori_look_down_angle: the original angle that camera looks down into the ground
        image_size: size of squared rendered image, default IMAGE_SIZE

    Returns:
        the rendered image
    """
    x, y = np.cos(ori_look_down_angle), np.sin(ori_look_down_angle)
    R1 = torch.cuda.FloatTensor([[[1, 0, 0], [0, x, y], [0, -y, x]]])
    x, y = np.cos(theta), np.sin(theta)
    mx, my, mz = phosa.get_verts_object().mean(dim=(0, 1)).detach().cpu().numpy()
    K = phosa.renderer.K
    if view == 'top':
        R2 = torch.cuda.FloatTensor([[[1, 0, 0], [0, x, -y], [0, y, x]]])
        t_cam_obj = torch.cuda.FloatTensor([[[mx], [my], [mz]]])
        t2 = torch.cuda.FloatTensor([[[0], [d], [d/2]]])
    elif view == 'left':
        R2 = torch.cuda.FloatTensor([[[x, 0, -y], [0, 1, 0], [y, 0, x]]])
        t_cam_obj = torch.cuda.FloatTensor([[[mx], [my], [mz]]])
        t2 = torch.cuda.FloatTensor([[[d], [0], [d/2]]])
    R = R2 @ R1
    t = R1@t_cam_obj + t2
    t = t.squeeze()
    renderer = nr.renderer.Renderer(
        image_size=image_size, K=K, R=R, t=t, orig_size=1
    )
    renderer.background_color = [1, 1, 1]
    renderer.light_direction = [1, 0.5, 1]
    renderer.light_intensity_direction = 0.3
    renderer.light_intensity_ambient = 0.5
    renderer.background_color = [1, 1, 1]
    rendered_img, _ = phosa.render(renderer)
    return rendered_img


def vis_3x3_grid(phosa,
                 seq,
                 frame,
                 kid,
                 object_masks,
                 objCD=0.,
                 bodyCD=0.,
                 img_cropped=None,
                 INPAINT=True,
                 behave_path="/data/data3/agavryushin/Datasets/behave/sequences/",
                 crop_base="/pub/scratch/huangd/behave_sam_crop",
                 text_descrip="",
                 save_path=None):
    """
    Args:
        seq: e.g. Date03_Sub03_plasticcontainer
        frame: e.g. t0007.000
        kid:: e.g. 2
        object_masks: object_parameters['masks'], torch.Tensor (1, image_size, image_size)
        objCD: the evaluated object chamfer distance
        bodyCD: the evaluated human chamfer distance
        img_cropped: if passed, will overwrite the img_cropped computed from cropping information
        INPAINT: whether in Inpainting mode
        behave_path: path to the behave dataset
        crop_base: path to crop information
        text_descrip: the center top title in the figure
        save_path: the path(s) to save the figure. None means no save.

    Returns:
        Visualization of a 3x3 grid plot, containing
        original image,      mask,                     cropped image
        top view,            rendered object mask,     rendered object on cropped image
        left view,           rendered person mask,     rendered person on cropped image

    """

    if os.path.isfile(seq):
        image = Image.open(seq)
        crop_info = {"x1": 0, "y1": 0, "x2": image.width, "y2": image.height}
    else:
        crop_info_path = crop_base + "/{}/{}/k{}.color_box.json".format(seq, frame, kid)
        with open(crop_info_path, "r") as f:
            crop_info = json.load(f)
        image = Image.open(behave_path + "/" + seq + "/" + frame + "/k{}.color.jpg".format(kid)).convert("RGB")

    fig, ax = plt.subplots(3, 3)
    ax[0, 0].imshow(np.array(image))
    rect = patches.Rectangle((crop_info['x1'], crop_info['y1']), crop_info['x2'] - crop_info['x1'],
                             crop_info['y2'] - crop_info['y1'],
                             linewidth=1, edgecolor='r', facecolor='none')
    ax[0, 0].add_patch(rect)
    ax[0, 0].set_title(f'input_img\n{image.size}', fontsize=10)
    if INPAINT:
        img_cropped = image.crop((int(crop_info["x1"]), int(crop_info["y1"]), int(crop_info["x2"]), int(crop_info["y2"])))
    assert object_masks.shape[0] == 1
    pred_mask = np.array(object_masks[0].detach().cpu())
    pred_mask = pred_mask[:512, :512]
    pred_mask = cv2.resize(pred_mask.astype(np.int64), dsize=(image_size, image_size), interpolation=cv2.INTER_NEAREST)
    ax[0, 1].imshow(pred_mask)
    ax[0, 1].set_title(f'input_mask\n{pred_mask.shape}', fontsize=10)
    w, h = img_cropped.size
    r = min(image_size / w, image_size / h)
    w = int(r * w)
    h = int(r * h)
    img_cropped = np.array(img_cropped.resize((w, h)))
    if img_cropped.max() > 1:
        img_cropped = img_cropped / 255.0
    ax[0, 2].imshow(img_cropped)
    ax[0, 2].set_title(f'resized_img\n{img_cropped.shape}', fontsize=10)

    # REND THE OBJECT
    verts_combined = phosa.get_verts_object()
    rend_obj, _, mask_obj = phosa.renderer.render(
        vertices=verts_combined, faces=phosa.faces_object, textures=phosa.textures_object
    )
    rend_obj = np.clip(rend_obj[0].detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
    rend_obj = rend_obj[:h, :w]
    ax[1, 0].imshow(render_view(phosa, view='top'))

    mask_obj = mask_obj[0].detach().cpu().numpy().astype(bool)
    mask_obj = mask_obj[:h, :w]
    ax[1, 1].imshow(mask_obj)

    h, w, c = img_cropped.shape
    new_image = img_cropped.copy()
    new_image[mask_obj] = rend_obj[mask_obj]
    new_image = (new_image * 255).astype(np.uint8)
    ax[1, 2].imshow(new_image)
    ax[1, 2].text(1.05, 0.5, f'obj_CD={objCD:.2f}', fontsize=10, va='center', rotation=-90,
                  transform=ax[1, 2].transAxes)

    # REND THE PERSON
    rend_person, _, mask_person = phosa.renderer.render(
        vertices=phosa.get_verts_person(), faces=phosa.faces_person, textures=phosa.textures_person
    )
    rend_person = np.clip(rend_person[0].detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
    rend_person = rend_person[:h, :w]
    ax[2, 0].imshow(render_view(phosa, view='left'))
    mask_person = mask_person[0].detach().cpu().numpy().astype(bool)
    mask_person = mask_person[:h, :w]
    ax[2, 1].imshow(mask_person)

    new_image = img_cropped.copy()
    new_image[mask_person] = rend_person[mask_person]
    new_image = (new_image * 255).astype(np.uint8)
    ax[2, 2].imshow(new_image)
    ax[2, 2].text(1.05, 0.5, f'person_CD={bodyCD:.2f}', fontsize=10, va='center', rotation=-90,
                  transform=ax[2, 2].transAxes)

    # image_name = seq + "/" + frame + f"/k{kid}.color.jpg"
    fig.suptitle(seq, fontsize=14, y=0.05)
    if text_descrip:
        fig.text(0.35, 0.965, text_descrip, color='red', fontsize='large')
    if save_path:
        if isinstance(save_path, list):
            for s_path in save_path:
                plt.savefig(s_path)
        else:
            assert isinstance(save_path, str)
            plt.savefig(save_path)
    plt.show()


def save_tensor_as_pcd(tensor: torch.Tensor, file_path: str):
    """
    Saves a PyTorch tensor as a PCD file.

    Args:
        tensor (torch.Tensor): Input tensor of shape (N, 3) representing the point cloud.
        file_path (str): Path to save the PCD file.
    """
    assert tensor.ndim == 2 and tensor.shape[1] == 3, "Input tensor must have shape (N, 3)."

    num_points = tensor.shape[0]

    # Header for PCD file
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {num_points}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {num_points}
DATA ascii"""

    # Convert tensor to numpy for easy saving
    point_data = tensor.numpy()

    # Write to the file
    with open(file_path, 'w') as f:
        f.write(header + "\n")
        for point in point_data:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"PCD file saved at: {file_path}")


def eval_align_both_models_from_sRt(person_vertices,
                                    s, R, t,
                                    cls,
                                    image_id_info,
                                    dataset,
                                    dataset_dir=None,
                                    align="person",
                                    chore_cd=True):
    if dataset == "behave":
        if dataset_dir in [None, ""]:
            dataset_dir = BEHAVE_DATASET_DIR

        obj_gt_origin_path = BEHAVE_GT_ORIGIN_PATH.format(cls=cls)
        cls_to_use = {"chairblack": "chair", "chairwood": "chair", "yogaball": "sports ball", "basketball": "sports ball"}.get(cls, cls)
        obj_gt_path = os.path.join(dataset_dir, "sequences", image_id_info["seq"], image_id_info["frame"], cls_to_use, "fit01", f"{cls_to_use}_fit.ply")
        person_gt_path = os.path.join(dataset_dir, "sequences", image_id_info["seq"], image_id_info["frame"], "person", "fit02", "person_fit.ply")
    elif dataset == "intercap":
        if dataset_dir in [None, ""]:
            dataset_dir = INTERCAP_DATASET_DIR

        id_sub = image_id_info["id_sub"]
        id_obj = image_id_info["id_obj"]
        id_seg = image_id_info["id_seg"]
        id_recon = image_id_info["id_recon"]
        obj_gt_origin_path = INTERCAP_GT_ORIGIN_PATH.format(cls=cls)  # glob.glob(obj_gt_origin_path_intercap.format(id_obj))[0]
        obj_gt_path = os.path.join(dataset_dir, "Res", id_sub, id_obj, f"Seg_{id_seg}", "Mesh", f"{id_recon}_second_obj.ply") # glob.glob(obj_gt_path_intercap.format(id_sub, id_obj, id_seg, id_recon))[0]
        person_gt_path = os.path.join(PROJECT_ROOT_DIR, "external", "intercap", "keyframes_smpl_gt", f'{id_sub}--{id_obj}--Seg_{id_seg}--Mesh--{id_recon}_second.obj')
    else:
        raise NotImplementedError(f'Evaluation not supported for dataset "{dataset}"')

    # load GT obj
    m = Mesh()
    m.load_from_file(obj_gt_path)
    obj_gt = m.v
    obj_gt = torch.tensor(obj_gt).float().unsqueeze(0)

    # load GT person
    m = Mesh()
    m.load_from_file(person_gt_path)
    person_gt = m.v
    person_gt = torch.tensor(person_gt).float().unsqueeze(0)

    # recover initial reconstruction from s, R, t
    m = Mesh()
    m.load_from_file(obj_gt_origin_path)
    obj_gt_origin = m.v

    # first center obj_gt_origin to align it with obj_gt_temp (verified via visualizations)
    m.v = m.v - m.v.mean(0)
    # get the same vertices as loaded from center_vertices(nr.load_obj(path))
    m.v -= m.v.min()
    m.v /= abs(m.v).max()
    m.v *= 2
    m.v -= m.v.max() / 2
    m.v = m.v - m.v.mean(0)
    m.v[:, 1] *= -1
    obj_gt_origin = m.v

    obj_pred = s * (np.matmul(obj_gt_origin, R) + t)
    recon = torch.tensor(np.concatenate([obj_pred, person_vertices], axis=0)).float().unsqueeze(0)
    obj_vert_num = recon.shape[1] - person_gt.shape[1]

    # alignment: recon -> gt
    if align == "both":
        bodyobj_aligned, scale, R, t = compute_similarity_transform_batch(recon.numpy(),
                                                                          torch.cat([obj_gt, person_gt], dim=-2).numpy())
    elif align == "person":
        body_aligned, scale, R, t = compute_similarity_transform_batch(recon[:, obj_vert_num:].numpy(),
                                                                       person_gt.numpy())
        obj_aligned = (scale[0] * R[0].dot(recon[0, :obj_vert_num].numpy().T) + t[0]).T
        bodyobj_aligned = np.concatenate((obj_aligned[None,...], body_aligned), 1)
    else:
        raise NotImplementedError()

    # compute chamfer distance
    if not chore_cd:
        body_dist_bi = chamfer_distance_torch(torch.tensor(bodyobj_aligned[:, obj_vert_num:]).float().cuda().contiguous(),
                                              person_gt.cuda().contiguous())[:2]
        body_dist = (body_dist_bi[0].mean() + body_dist_bi[1].mean()) / 2
        obj_dist_bi = chamfer_distance_torch(torch.tensor(bodyobj_aligned[:, :obj_vert_num]).float().cuda().contiguous(),
                                           obj_gt.cuda().contiguous())[:2]
        obj_dist = (obj_dist_bi[0].mean() + obj_dist_bi[1].mean()) / 2
        body_dist = body_dist.sqrt().item()
        obj_dist = obj_dist.sqrt().item()
    else:
        body_dist = chamfer_distance_chore(np.array(person_gt[0]), np.array(bodyobj_aligned[:, obj_vert_num:][0]))
        obj_dist = chamfer_distance_chore(np.array(obj_gt)[0], np.array(bodyobj_aligned[:, :obj_vert_num][0]))
    return bodyobj_aligned, body_dist, obj_dist

