import os
import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(1, project_root)

import joblib
import json
import neural_renderer as nr
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.ndimage.morphology import distance_transform_edt
import torch
import torch.nn as nn
import time
from tqdm.auto import tqdm

from config.defaults import (
    ADAPTIVE_ITERS_CONVERGE_THRESH,
    ADAPTIVE_ITERS_STOPPING_TOLERANCE,
    ADAPTIVE_ITERS_PRUNE_CANDIDATES,
    ADAPTIVE_ITERS_PRUNE_RATE,
    BBOX_EXPANSION_FACTOR,
    DEFAULT_IMAGE_SIZE,
    FOCAL_LENGTH,
    REND_SIZE,
    RENDERER,
    DEFAULT_USE_ADAPTIVE_ITERS,
    DEFAULT_USE_ADAPTIVE_ITERS_INIT,
    USE_OPTUNA
)
from utils import (
    PerspectiveRenderer,
    center_vertices,
    compute_K_roi,
    compute_random_rotations,
    compute_random_rotations_optuna,
    crop_image_with_bbox,
    matrix_to_rot6d,
    rot6d_to_matrix,
)
from utils.camera import local_to_global_cam
from utils.pointrend import get_class_masks_from_instances
if "kaolin" in RENDERER:
    import kaolin as kal
if USE_OPTUNA:
    import optuna


class PoseOptimizer(nn.Module):
    """
    Computes the optimal object pose from an instance mask and an exemplar mesh. We
    optimize an occlusion-aware silhouette loss that consists of a one-way chamfer loss
    and a silhouette matching loss.
    """

    def __init__(
        self,
        ref_image,
        vertices,
        faces,
        textures,
        rotation_init,
        translation_init,
        batch_size=1,
        kernel_size=7,
        K=None,
        power=0.25,
        lw_chamfer=0.5,
        depth_mask=None,
        inverse_depth=False,
        lw_depth=0.,
        normal_map=None,
        lw_normal=0.
    ):
        assert ref_image.shape[0] == ref_image.shape[1], "Must be square."
        super(PoseOptimizer, self).__init__()

        vertices_batch = vertices.repeat(batch_size, 1, 1)
        faces_batch = faces.repeat(batch_size, 1, 1)
        self.register_buffer("vertices", vertices_batch)
        self.register_buffer("faces", faces_batch)
        self.register_buffer("textures", textures.repeat(batch_size, 1, 1, 1, 1, 1))

        # Load reference mask.
        # Convention for silhouette-aware loss: -1=occlusion, 0=bg, 1=fg.
        image_ref = torch.from_numpy((ref_image > 0).astype(np.float32))
        keep_mask = torch.from_numpy((ref_image >= 0).astype(np.float32))
        self.register_buffer("image_ref", image_ref.repeat(batch_size, 1, 1))
        self.register_buffer("keep_mask", keep_mask.repeat(batch_size, 1, 1))
        self.use_depth_loss = False
        self.use_normal_loss = False
        self.lw_normal = lw_normal
        if normal_map is not None and lw_normal > 0.:
            self.use_normal_loss = True
            self.register_buffer("normal_map", normal_map * (normal_map > -1.0) * self.image_ref[0].unsqueeze(-1))

        if depth_mask is not None:
            self.lw_depth = lw_depth
            if self.lw_depth > 0.:
                self.use_depth_loss = True

            if inverse_depth:
                depth_mask = depth_mask.clone()
                depth_mask[depth_mask > 0] = 1.0 / depth_mask[depth_mask > 0]

            # remove accidentally segmented background
            binary_from_depth = binary_erosion((depth_mask[0].numpy() * self.image_ref[0].numpy()), iterations=10)
            eroded = depth_mask * self.image_ref - depth_mask * self.image_ref * binary_from_depth
            qtl = torch.quantile((depth_mask * self.image_ref)[depth_mask * self.image_ref > 0].flatten(), q=0.9) if (depth_mask * self.image_ref).sum() > 0 else eroded[0].max() + 1
            depth_mask[:, eroded[0] > qtl] = 0


            self.register_buffer("depth_mask", depth_mask.repeat(batch_size, 1, 1))
            self.register_buffer("ref_depth_mask", self.depth_mask * self.image_ref * binary_from_depth)

            nrmzd_ref_depth_mask = self.ref_depth_mask.clone()[0].unsqueeze(0)
            self.register_buffer("nrmzd_ref_depth_mask", nrmzd_ref_depth_mask)
            if len(self.ref_depth_mask[self.ref_depth_mask > 0]) > 0:
                min_depth = self.ref_depth_mask[self.ref_depth_mask > 0].min() if (self.ref_depth_mask > 0).sum() > 0 else 0.0
                max_depth = self.ref_depth_mask[self.ref_depth_mask > 0].max() if (self.ref_depth_mask > 0).sum() > 0 else 1e-6
                B=1
                nrmzd_ref_depth_mask[nrmzd_ref_depth_mask > 0] = \
                    (nrmzd_ref_depth_mask[nrmzd_ref_depth_mask > 0] - min_depth) / (max_depth - min_depth)
                # denoise the normalized depth map
                sorted_nrmzd_ref_depth_mask = torch.sort(self.nrmzd_ref_depth_mask.view(B, -1), dim=-1).values
                num_lt_0 = (sorted_nrmzd_ref_depth_mask <= 0).sum(-1)  # (B,) index of <= 0 values per batch sample
                num_gt_0 = (sorted_nrmzd_ref_depth_mask > 0).sum(-1)  # (B,) index of > 0 values per batch sample
                quantile01 = sorted_nrmzd_ref_depth_mask[
                    torch.arange(B).unsqueeze(-1), int(num_lt_0 + num_gt_0 * 0.01)]  # (B,1)
                quantile99 = sorted_nrmzd_ref_depth_mask[
                    torch.arange(B).unsqueeze(-1), int(num_lt_0 + num_gt_0 * 0.99)]  # (B,1)
                renorm01 = quantile01 > 0.2  # (B, 1) whether to re-normalization per batch sample
                self.nrmzd_ref_depth_mask[self.nrmzd_ref_depth_mask < quantile01] = \
                    0 * renorm01 + self.nrmzd_ref_depth_mask[self.nrmzd_ref_depth_mask < quantile01] * (~renorm01)
                renorm99 = quantile99 < 0.8  # (B, 1) whether to re-normalization per batch sample
                self.nrmzd_ref_depth_mask[self.nrmzd_ref_depth_mask > quantile99] = \
                    0 * renorm99 + self.nrmzd_ref_depth_mask[self.nrmzd_ref_depth_mask > quantile99] * (~renorm99)
                min_depth = self.nrmzd_ref_depth_mask[self.nrmzd_ref_depth_mask > 0].min() if (self.nrmzd_ref_depth_mask > 0).sum() > 0 else 0.0
                max_depth = self.nrmzd_ref_depth_mask[self.nrmzd_ref_depth_mask > 0].max() if (self.nrmzd_ref_depth_mask > 0).sum() > 0 else 1e-6
                self.nrmzd_ref_depth_mask[self.nrmzd_ref_depth_mask > 0] = \
                    (self.nrmzd_ref_depth_mask[self.nrmzd_ref_depth_mask > 0] - min_depth) / (max_depth - min_depth)
            self.nrmzd_ref_depth_mask = self.nrmzd_ref_depth_mask.repeat(batch_size, 1, 1)
        self.pool = torch.nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=(kernel_size // 2)
        )
        self.rotations = nn.Parameter(rotation_init.clone().float(), requires_grad=True)
        if rotation_init.shape[0] != translation_init.shape[0]:
            translation_init = translation_init.repeat(batch_size, 1, 1)
        self.translations = nn.Parameter(
            translation_init.clone().float(), requires_grad=True
        )
        mask_edge = self.compute_edges(image_ref.unsqueeze(0)).cpu().numpy()
        edt = distance_transform_edt(1 - (mask_edge > 0)) ** (power * 2)
        self.register_buffer(
            "edt_ref_edge", torch.from_numpy(edt).repeat(batch_size, 1, 1).float()
        )
        # Setup renderer.
        if K is None:
            K = torch.cuda.FloatTensor([[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]])
        R = torch.eye(3).unsqueeze(0).cuda()
        t = torch.zeros(1, 3).cuda()

        self.num_faces = faces.shape[0]
        self.batch_size = batch_size
        self.K = K
        self.R = R
        self.t = t
        self.cam_proj = torch.cuda.FloatTensor([[1.0], [1.0], [-1.0]])
        self.cam_rot = R.cuda().repeat(self.batch_size, 1, 1)
        self.cam_trans = t.cuda().repeat(self.batch_size, 1)
        self.image_size = ref_image.shape[0]

        # needed for final visualization rendering
        self.renderer = nr.renderer.Renderer(
            image_size=ref_image.shape[0],
            K=K,
            R=R,
            t=t,
            orig_size=1,
            anti_aliasing=False,
        )
        self.lw_chamfer = lw_chamfer

    def apply_transformation(self):
        """
        Applies current rotation and translation to vertices.
        """
        rots = rot6d_to_matrix(self.rotations)
        return torch.matmul(self.vertices, rots) + self.translations

    def compute_offscreen_loss(self, verts):
        """
        Computes loss for offscreen penalty. This is used to prevent the degenerate
        solution of moving the object offscreen to minimize the chamfer loss.
        """
        # On-screen means xy between [-1, 1] and far > depth > 0
        proj = nr.projection(
            verts,
            self.K,
            self.R,
            self.t,
            self.renderer.dist_coeffs, 
            orig_size=1,
        )
        xy, z = proj[:, :, :2], proj[:, :, 2:]
        zeros = torch.zeros_like(z)
        lower_right = torch.max(xy - 1, zeros).sum(dim=(1, 2))  # Amount greater than 1
        upper_left = torch.max(-1 - xy, zeros).sum(dim=(1, 2))  # Amount less than -1
        behind = torch.max(-z, zeros).sum(dim=(1, 2))
        too_far = torch.max(z - self.renderer.far, zeros).sum(dim=(1, 2))
        return lower_right + upper_left + behind + too_far

    def compute_edges(self, silhouette):
        return self.pool(silhouette) - silhouette

    def forward(self):
        t1 = time.time()
        verts = self.apply_transformation()  # verts = self.vertices + self.translations

        if "kaolin" in RENDERER:
            vertices_batch = verts
            faces_batch = self.faces
            vertices_to_faces = nr.vertices_to_faces(vertices_batch, faces_batch)
            
            face_features = torch.zeros((self.batch_size, self.num_faces, 3, 3), device=vertices_to_faces.device)
            for i in range(3):
                face_features[:, :, i, i] = 1
            faces = self.faces[0].long()

            vertices_camera = kal.render.camera.rotate_translate_points(vertices_batch, self.cam_rot,
                                                                        self.cam_trans)
            # we do not need to scale by the depth
            vertices_image_depth = nr.projection(vertices_batch, self.K, self.R, self.t, self.renderer.dist_coeffs, 1)  # keep orig_size to 1!
            vertices_image = vertices_image_depth[:, :, :2]  # / vertices_image_depth[:, :, 2:3]

            face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(vertices_camera, faces)
            face_vertices_image = kal.ops.mesh.index_vertices_by_faces(vertices_image, faces)

            face_vertices = face_vertices_camera * torch.tensor([-1, +1, +1], device=face_vertices_camera.device)

            edges_dist0 = face_vertices[:, :, 1] - face_vertices[:, :, 0]
            edges_dist1 = face_vertices[:, :, 2] - face_vertices[:, :, 0]
            face_normals = torch.cross(edges_dist0, edges_dist1, dim=2)

            face_normals_length = face_normals.norm(dim=2, keepdim=True)
            face_normals = face_normals / (face_normals_length + 1e-10)


            rendered_features, rendered_face_idx =\
                kal.render.mesh.rasterize(self.image_size, self.image_size, -vertices_to_faces[:, :, :, -1],
                                          face_vertices_image, face_features)
            soft_mask = kal.render.mesh.dibr_soft_mask(face_vertices_image, rendered_face_idx, sigmainv=1e7)
            image = self.keep_mask * soft_mask
            rendered_face_idx2 = rendered_face_idx.masked_fill(rendered_face_idx == -1, faces.shape[0])
            if self.use_depth_loss:
                depth_vertices = vertices_image_depth[:, :, 2]  # (B, N_vertices,)
                B = depth_vertices.shape[0]
                t2 = time.time()
                depth_faces = torch.gather(depth_vertices.unsqueeze(-1).expand(-1, -1, 3), 1, faces.unsqueeze(0).expand(B, -1, -1))
                depth_min_per_batch = torch.min(depth_faces, dim=-2, keepdim=True).values.min(dim=-1, keepdim=True).values.repeat(1, 1, 3) * 0.95
                t3 = time.time()
                depth_faces = torch.cat((depth_faces, depth_min_per_batch), dim=-2)
                t4 = time.time()
                rendered_face_idx2 = rendered_face_idx.masked_fill(rendered_face_idx == -1, faces.shape[0])
                depth_rend = (rendered_features[..., 0] * torch.gather(depth_faces[:, :, 0].unsqueeze(-1).expand(-1, -1, rendered_face_idx2.shape[-1]), 1,rendered_face_idx2)
                              + rendered_features[..., 1] * torch.gather(depth_faces[:, :, 1].unsqueeze(-1).expand(-1, -1,rendered_face_idx2.shape[-1]), 1,rendered_face_idx2)
                              + rendered_features[..., 2] * torch.gather(depth_faces[:, :, 2].unsqueeze(-1).expand(-1, -1,rendered_face_idx2.shape[-1]), 1,rendered_face_idx2))
                t5 = time.time()
                min_depth = torch.min(depth_rend.view(B, -1), dim=1).values.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1) the minimum depth, representing background
                mask_intersec = (self.nrmzd_ref_depth_mask > 0) * (
                            depth_rend > min_depth)  # this is the area where we will compute depth loss
                t6 = time.time()

                bg_mask = depth_rend <= min_depth
                bg_mask_non_intersec = torch.logical_or(bg_mask, torch.logical_not(mask_intersec))
                depth_rend_ = depth_rend.masked_fill(bg_mask_non_intersec, float('inf'))  # bg_mask

                min_dep = torch.min(depth_rend_.view(B, -1), dim=1).values.unsqueeze(-1).unsqueeze(-1)  # the min depth, excluding background
                depth_rend_.masked_fill_(bg_mask_non_intersec, float('-inf'))  # bg_mask
                max_dep = torch.max(depth_rend_.view(B, -1), dim=1).values.unsqueeze(-1).unsqueeze(-1)  # the max depth, excluding background
                nrmzed_depth_rend = ((depth_rend - min_dep) / (max_dep - min_dep + 1e-6))
                nrmzed_depth_rend = nrmzed_depth_rend.masked_fill(bg_mask_non_intersec, 0.)  # bg_mask
                t7 = time.time()
            
            if self.use_normal_loss:
                # zero_normals needed for unoccupied pixels
                zero_normals = torch.zeros(face_normals.shape[0], 1, face_normals.shape[2], device=face_normals.device)
                face_normals = torch.cat((face_normals, zero_normals), 1)

                mesh_normals_x = face_normals[:, :, 0]
                mesh_normals_y = face_normals[:, :, 1]
                mesh_normals_z = face_normals[:, :, 2]

                rend_normals_x = torch.gather(mesh_normals_x, 1, rendered_face_idx2.view((rendered_face_idx2.shape[0], -1))).view(rendered_face_idx2.shape)
                rend_normals_y = torch.gather(mesh_normals_y, 1, rendered_face_idx2.view((rendered_face_idx2.shape[0], -1))).view(rendered_face_idx2.shape)
                rend_normals_z = torch.gather(mesh_normals_z, 1, rendered_face_idx2.view((rendered_face_idx2.shape[0], -1))).view(rendered_face_idx2.shape)
                normal_rend_check = torch.stack((rend_normals_x, rend_normals_y, rend_normals_z), dim=-1)  # [..., [1, 0, 2]]  (normal_vis_1) nothing (normal_vis_2)

                normal_rend = torch.gather(face_normals.unsqueeze(2).expand(-1, -1, rendered_face_idx2.shape[-1], -1),
                                           1, rendered_face_idx2.unsqueeze(-1).expand(-1, -1, -1, 3)) * torch.tensor((1.0, 1.0, 1.0)).cuda()  # [..., [1, 0, 2]]  (normal_vis_1)
                #normal_rend = - normal_rend
                normal_loss_compute_area = (self.normal_map != 0).any(-1).unsqueeze(0) * (normal_rend != 0).any(-1)
        else:
            image = self.keep_mask * self.renderer(verts, self.faces, mode="silhouettes")
            if self.use_depth_loss:
                assert False, 'depth loss is only supported when kaolin=True'

        loss_dict = {}
        loss_dict["mask"] = torch.sum((image - self.image_ref) ** 2, dim=(1, 2))

        if self.use_depth_loss:
            depth_loss_sum = torch.sum((((nrmzed_depth_rend - self.nrmzd_ref_depth_mask)*mask_intersec) + (image - self.image_ref))**2, dim=(1, 2))
            loss_dict["depth"] = depth_loss_sum * self.lw_depth
        
        if self.use_normal_loss:
            # the closer to 1, the better:
            dot_prod = (normal_rend * self.normal_map.unsqueeze(0)).sum(-1)
            normal_mismatch = normal_loss_compute_area*(1.0 - (0.5 + dot_prod * 0.5))  # torch.abs
            image_mismatch = (~normal_loss_compute_area)*(image - self.image_ref)  # causes bad performance drop: torch.nn.functional.relu(self.image_ref - image)
            normal_loss_sum = torch.sum((normal_mismatch + image_mismatch)**2, dim=(1, 2))
            loss_dict['normal'] = normal_loss_sum * self.lw_normal

            loss_dict["save_normal_mismatch"] = normal_mismatch.sum(dim=(1, 2)).detach()
            loss_dict["save_image_mismatch"] = torch.abs(image_mismatch).sum(dim=(1, 2)).detach()

        
        loss_dict["save_image_area"] = image.sum(dim=(1, 2)).detach()
        loss_dict["save_ref_area"] = self.image_ref.sum(dim=(1, 2)).detach()
        loss_dict["save_image_iou"] = (image.sum(dim=(1, 2)) / (1e-6 + torch.logical_or(self.image_ref > 0, image > 0).sum(dim=(1, 2)))).detach()
        
        loss_dict["chamfer"] = self.lw_chamfer * torch.sum(
            self.compute_edges(image) * self.edt_ref_edge, dim=(1, 2)
        )
        loss_dict["offscreen"] = 1000 * self.compute_offscreen_loss(verts)
        t8 = time.time()
        # print(f't1:{t2 - t1:.3f} '  # uncomment to show the time
        #       f't2:{t3 - t2:.3f} '
        #       f't3:{t4 - t3:.3f} '
        #       f't4:{t5 - t4:.3f} '
        #       f't5:{t6 - t5:.3f} '
        #       f't6:{t7 - t6:.3f} '
        #       f't7:{t8 - t7:.3f} '
        #       f'total{t8 - t1: .3f}')

        losses = sum([v for k, v in loss_dict.items() if not k.startswith("save_")])
        ind = torch.argmin(losses)
        
        if self.use_normal_loss:
            os.makedirs("normal_vis", exist_ok=True)
            Image.fromarray(((0.5 + normal_rend[ind] * torch.tensor((-0.5, 0.5, 0.5)).cuda()).detach().cpu().numpy() * 255).astype(np.uint8)).save(f"normal_vis/{int(t1*1000)}_normal_rend_{ind}.png")
            Image.fromarray(((0.5 + self.normal_map * 0.5).detach().cpu().numpy() * 255).astype(np.uint8)).save(f"normal_vis/{int(t1*1000)}_normal_map_{ind}.png")
            Image.fromarray((rendered_face_idx2[ind].detach().cpu().numpy()).astype(np.uint8)).save(f"normal_vis/{int(t1*1000)}_faces_{ind}.png")

        if self.use_depth_loss:
            os.makedirs("depth_vis", exist_ok=True)
            Image.fromarray((nrmzed_depth_rend[ind].detach().cpu().numpy() * 255).astype(np.uint8)).save(f"depth_vis/nrmzed_depth_rend_{int(t1*1000)}_depth_{ind}.png")
            Image.fromarray((self.nrmzd_ref_depth_mask[ind].detach().cpu().numpy() * 255).astype(np.uint8)).save(f"depth_vis/nrmzed_depth_rend_{int(t1*1000)}_depth_ref_{ind}.png")
            Image.fromarray((rendered_face_idx2[ind].detach().cpu().numpy()).astype(np.uint8)).save(f"depth_vis/nrmzed_depth_rend_{int(t1*1000)}_faces_{ind}.png")

        return loss_dict, image

    def render(self):
        """
        Renders objects according to current rotation and translation.
        """
        verts = self.apply_transformation()
        images = self.renderer(verts, self.faces, torch.tanh(self.textures))[0]
        images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
        return images


def compute_bbox_proj(verts, f, img_size=256):
    """
    Computes the 2D bounding box of vertices projected to the image plane.

    Args:
        verts (B x N x 3): Vertices.
        f (float): Focal length.
        img_size (int): Size of image in pixels.

    Returns:
        Bounding box in xywh format (Bx4).
    """
    xy = verts[:, :, :2]
    z = verts[:, :, 2:]
    proj = f * xy / z + 0.5  # [0, 1]
    proj = proj * img_size  # [0, img_size]
    u, v = proj[:, :, 0], proj[:, :, 1]
    x1, x2 = u.min(1).values, u.max(1).values
    y1, y2 = v.min(1).values, v.max(1).values
    return torch.stack((x1, y1, x2 - x1, y2 - y1), 1)


def compute_optimal_translation(bbox_target, vertices, f=1, img_size=256):
    """
    Computes the optimal translation to align the mesh to a bounding box using
    least squares.

    Args:
        bbox_target (list): bounding box in xywh.
        vertices (B x V x 3): Batched vertices.
        f (float): Focal length.
        img_size (int): Image size in pixels.

    Returns:
        Optimal 3D translation (B x 3).
    """
    bbox_mask = np.array(bbox_target)
    mask_center = bbox_mask[:2] + bbox_mask[2:] / 2
    diag_mask = np.sqrt(bbox_mask[2] ** 2 + bbox_mask[3] ** 2)
    B = vertices.shape[0]
    x = torch.zeros(B).cuda()
    y = torch.zeros(B).cuda()
    z = 2.5 * torch.ones(B).cuda()
    for _ in range(50):
        translation = torch.stack((x, y, z), -1).unsqueeze(1)
        v = vertices + translation
        bbox_proj = compute_bbox_proj(v, f=1, img_size=img_size)
        diag_proj = torch.sqrt(torch.sum(bbox_proj[:, 2:] ** 2, 1))
        delta_z = z * (diag_proj / diag_mask - 1)
        z = z + delta_z
        proj_center = bbox_proj[:, :2] + bbox_proj[:, 2:] / 2
        x += (mask_center[0] - proj_center[:, 0]) * z / f / img_size
        y += (mask_center[1] - proj_center[:, 1]) * z / f / img_size
    return torch.stack((x, y, z), -1).unsqueeze(1)


def find_optimal_pose(
    vertices,
    faces,
    mask,
    bbox,
    square_bbox,
    image_size,
    batch_size=170,
    num_iterations=50,
    num_initializations=2000,
    lr=2e-3,
    depth_mask=None,
    inverse_depth=False,
    lw_depth=0.,
    normal_map=None,
    lw_normal=0.,
    use_adaptive_iters=DEFAULT_USE_ADAPTIVE_ITERS
):
    # "bbox" contains bbox tailored to object
    ts = 1
    textures = torch.ones(faces.shape[0], ts, ts, ts, 3, dtype=torch.float32).cuda()
    x, y, b, _ = square_bbox
    L = max(image_size)
    K_roi = compute_K_roi((x, y), b, L, focal_length=FOCAL_LENGTH)
    best_losses = np.inf
    best_rots = None
    best_trans = None
    best_loss_save_single = {}
    best_loss_single = np.inf
    best_rots_single = None
    best_trans_single = None
    best_sil_single = None
    loop = tqdm(total=np.ceil(num_initializations / batch_size) * num_iterations)

    if USE_OPTUNA:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler())

    for _ in range(0, num_initializations, batch_size):
        model = None
        for optim_step_idx in [0]: # [0, 1]:
            if optim_step_idx == 0:
                if USE_OPTUNA:
                    trial_numbers = []
                    rotations_init = compute_random_rotations_optuna(study, trial_numbers, batch_size, upright=False)
                else:
                    rotations_init = compute_random_rotations(batch_size, upright=False)

                translations_init = compute_optimal_translation(
                    bbox_target=np.array(bbox) * REND_SIZE / L,
                    vertices=torch.matmul(vertices.unsqueeze(0), rotations_init),
                    img_size=REND_SIZE  # added 250105
                )
                
                model = PoseOptimizer(
                    ref_image=mask,
                    vertices=vertices,
                    faces=faces,
                    textures=textures,
                    rotation_init=matrix_to_rot6d(rotations_init),
                    translation_init=translations_init,
                    batch_size=batch_size,
                    K=K_roi,
                    depth_mask=depth_mask,
                    inverse_depth=inverse_depth,
                    lw_depth=lw_depth,
                    normal_map=normal_map,
                    lw_normal=lw_normal
                    #lw_chamfer=1.0
                )
                model.cuda()
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            else:
                model.lw_depth = lw_depth
                model.use_depth_loss = True

            best_loss_this_batch = np.inf
            no_better_accumu = 0  # accumulate how many iterations have passed since last best loss, in this batch
            best_loss_batch = np.inf * torch.ones(batch_size).cuda()
            for num_iter in range(num_iterations):
                no_better_accumu += 1
                optimizer.zero_grad()
                loss_dict, sil = model()
                losses = sum([v for k, v in loss_dict.items() if not k.startswith("save_")])
                loss = losses.sum()
                loss.backward()
                optimizer.step()
                best_loss_batch = torch.minimum(best_loss_batch, losses)
                if losses.min() < best_loss_single:
                    ind = torch.argmin(losses)
                    best_loss_single = losses[ind]
                    best_rots_single = model.rotations[ind].detach().clone()
                    best_trans_single = model.translations[ind].detach().clone()
                    best_sil_single = sil[ind].detach().cpu().numpy().astype(np.uint8) * 255
                    best_loss_save_single = {k: v[ind] for k, v in loss_dict.items() if k.startswith("save_")}

                if losses.min() < best_loss_this_batch - ADAPTIVE_ITERS_CONVERGE_THRESH:
                    ind = torch.argmin(losses)
                    best_loss_this_batch = losses.min()
                    no_better_accumu = 0
                    info = ''
                    for k in sorted([*loss_dict.keys()]):
                        info += k
                        info += f": {loss_dict[k][ind].data:.2f}  "
                    loop.set_description(('[init] best loss: {best_loss_this_batch:.2f}  '.format(best_loss_this_batch=best_loss_this_batch)+info).strip())

                loop.update()

            if use_adaptive_iters:
                print(f'\nAfter {num_iter} iterations, there have passed {no_better_accumu} iters since last best loss.')

                if ADAPTIVE_ITERS_PRUNE_CANDIDATES is True and optim_step_idx == 1:
                    keep_inds = torch.argsort(losses.detach())[:int(batch_size*ADAPTIVE_ITERS_PRUNE_RATE)]
                    model.vertices = model.vertices[keep_inds]
                    model.faces = model.faces[keep_inds]
                    model.textures = model.textures[keep_inds]
                    model.image_ref = model.image_ref[keep_inds]
                    model.keep_mask = model.keep_mask[keep_inds]
                    model.edt_ref_edge = model.edt_ref_edge[keep_inds]
                    model.rotations = nn.Parameter(model.rotations[keep_inds])
                    model.translations = nn.Parameter(model.translations[keep_inds])
                    model.batch_size = len(keep_inds)
                    model.cam_rot = model.cam_rot[keep_inds]
                    model.cam_trans = model.cam_trans[keep_inds]
                    optimizer.param_groups[0]['params'][0] = model.rotations
                    optimizer.param_groups[0]['params'][1] = model.translations
                    losses = torch.sort(losses).values[:model.batch_size]

                while no_better_accumu <= ADAPTIVE_ITERS_STOPPING_TOLERANCE:  # Continue optimization until convergence
                    num_iter += 1
                    no_better_accumu += 1
                    optimizer.zero_grad()
                    loss_dict, sil = model()
                    losses = sum([v for k, v in loss_dict.items() if not k.startswith("save_")])
                    loss = losses.sum()
                    loss.backward()
                    optimizer.step()
                    if ADAPTIVE_ITERS_PRUNE_CANDIDATES is True:
                        best_loss_batch[keep_inds] = torch.minimum(best_loss_batch[keep_inds], losses)
                    else:
                        best_loss_batch = torch.minimum(best_loss_batch, losses)

                    if losses.min() < best_loss_single:
                        ind = torch.argmin(losses)
                        best_loss_single = losses[ind]
                        best_rots_single = model.rotations[ind].detach().clone()
                        best_trans_single = model.translations[ind].detach().clone()
                        best_loss_save_single = {k: v[ind] for k, v in loss_dict.items() if k.startswith("save_")}

                    if losses.min() < best_loss_this_batch - ADAPTIVE_ITERS_CONVERGE_THRESH:
                        ind = torch.argmin(losses)
                        best_loss_this_batch = losses.min()
                        no_better_accumu = 0
                        info = ''
                        for k in loss_dict:
                            info += k
                            info += f" {loss_dict[k][ind].data:.2f}  "

        if best_rots is None:
            best_rots = model.rotations
            best_trans = model.translations
            best_losses = losses
        else:
            best_rots = torch.cat((best_rots, model.rotations), 0)
            best_trans = torch.cat((best_trans, model.translations), 0)
            best_losses = torch.cat((best_losses, losses))
        
        inds = torch.argsort(best_losses)
        best_losses = best_losses[inds][:model.batch_size].detach().clone()
        best_trans = best_trans[inds][:model.batch_size].detach().clone()
        best_rots = best_rots[inds][:model.batch_size].detach().clone()

        if USE_OPTUNA:
            for trial_number, loss in zip(trial_numbers, best_loss_batch):
                study.tell(trial_number, loss.detach().cpu().item())

    loop.close()
    # Add best ever:
    best_rotations = torch.cat((best_rots_single.unsqueeze(0), best_rots[:-1]), 0)
    best_translations = torch.cat((best_trans_single.unsqueeze(0), best_trans[:-1]), 0)
    model.rotations = nn.Parameter(best_rotations)
    model.translations = nn.Parameter(best_translations)

    return model, best_loss_save_single


def visualize_optimal_poses(model, image_crop, mask, score=0):
    """
    Visualizes the 8 best-scoring object poses.

    Args:
        model (PoseOptimizer).
        image_crop (H x H x 3).
        mask (M x M x 3).
        score (float): Mask confidence score (optional).
    """
    num_vis = 8
    rotations = model.rotations
    translations = model.translations
    verts = model.vertices[0]
    faces = model.faces[0]
    loss_dict, sil = model()
    losses = sum(loss_dict.values())
    K_roi = model.renderer.K
    inds = torch.argsort(losses)[:num_vis]
    obj_renderer = PerspectiveRenderer()

    img_array = []
    img_array.append(image_crop)  # 141 x 141
    img_array.append(mask)  # 256 x 256
    scores = []

    for i, ind in enumerate(inds.cpu().numpy()):
        img_array.append(
            obj_renderer(
                vertices=verts,
                faces=faces,
                image=image_crop,
                translation=translations[ind],
                rotation=rot6d_to_matrix(rotations)[ind],
                color_name="red",
                K=K_roi,
            )
        )
        scores.append(losses[ind].item())

    return [img_array, scores]


def find_initial_object_poses(
    instances,
    vertices=None,
    faces=None,
    class_id=None,
    class_name=None,
    mesh_index=0,
    visualize=False,
    image=None,
    batch_size=512,
    num_iterations=50,
    num_initializations=2000,
    st=None,
    mesh_path=None,
    depth_mask=None,
    inverse_depth=False,
    lw_depth=0.,
    normal_map=None,
    lw_normal=0.,
    use_adaptive_iters=DEFAULT_USE_ADAPTIVE_ITERS
):
    """
    Optimizes for pose with respect to a target mask using an occlusion-aware silhouette
    loss.

    Args:
        instances: PointRend or Mask R-CNN instances.
        vertices (N x 3): Mesh vertices (If not set, loads vertices using class name
            and mesh index).
        faces (F x 3): Mesh faces (If not set, loads faces using class name and mesh
            index).
        class_id (int): Class index if no vertices/faces nor class name is given.
        class_name (str): Name of class.
        mesh_index (int): Mesh index for classes with multiple mesh models.
        visualize (bool): If True, visualizes the top poses found.
        image (H x W x 3): Image used for visualization.
        depth_mask (1 x 256 x 256): Reference depth mask used for depth loss

    Returns:
        dict {str: torch.cuda.FloatTensor}
            rotations (N x 3 x 3): Top rotation matrices.
            translations (N x 1 x 3): Top translations.
            target_masks (N x 256 x 256): Cropped occlusion-aware masks (for silhouette
                loss).
            masks (N x image_size x image_size): Object masks (for depth ordering loss).
            K_roi (N x 3 x 3): Camera intrinsics corresponding to each object ROI crop.
    """
    assert class_id is not None
    assert vertices is None and faces is None
    vertices, faces = nr.load_obj(mesh_path)
    vertices, faces = center_vertices(vertices, faces)
    # print("# vertices: ", len(vertices))
    # print("# faces: ", len(faces))

    class_masks, annotations = get_class_masks_from_instances(
        instances=instances,
        class_id=class_id,
        add_ignore=True,
        rend_size=REND_SIZE,
        bbox_expansion=BBOX_EXPANSION_FACTOR,
        min_confidence=0.90,
    )

    object_parameters = {
        "rotations": [],
        "translations": [],
        "target_masks": [],
        "K_roi": [],
        "masks": []
    }

    image_size = (image.shape[-1], image.shape[-2])

    vis_res = []
    for _, (mask, annotation) in enumerate(zip(class_masks, annotations)):
        model, best_loss_save_single = find_optimal_pose(
            vertices=vertices,
            faces=faces,
            mask=mask,
            bbox=annotation["bbox"],
            square_bbox=annotation["square_bbox"],
            image_size=image_size,
            batch_size=batch_size,
            num_iterations=num_iterations,
            num_initializations=num_initializations,
            depth_mask=depth_mask,
            inverse_depth=inverse_depth,
            lw_depth=lw_depth,
            normal_map=normal_map,
            lw_normal=lw_normal,
            use_adaptive_iters=use_adaptive_iters
        )
        if visualize:
            if image is None:
                image = np.zeros(image_size, dtype=np.uint8)
            img_array, scores = visualize_optimal_poses(
                model=model,
                image_crop=crop_image_with_bbox(image, annotation["square_bbox"]),
                mask=mask,
                score=annotation["score"],
            )
            vis_res.append([img_array, scores])

        object_parameters["rotations"].append(
            rot6d_to_matrix(model.rotations)[0].detach()
        )
        object_parameters["translations"].append(model.translations[0].detach())
        object_parameters["target_masks"].append(torch.from_numpy(mask).cuda())
        object_parameters["K_roi"].append(model.K.detach())
        object_parameters["masks"].append(annotation["mask"].cuda())

        for k, v in best_loss_save_single.items():
            if k not in object_parameters:
                object_parameters[k] = []
            object_parameters[k].append(v)

        break

    for k, v in object_parameters.items():
        if len(v):
            object_parameters[k] = torch.stack(v)
        else:
            # print(vis_res)
            return None, vis_res

    if visualize:
        return object_parameters, vis_res
    else:
        return object_parameters, None


def estimate_init_object_pose(selected_obj_name, cur_imname, instances, class_name, image,
                              num_init_iters, output_class_dir, rank, additional_name, dataset,
                              class_id=-1, overwrite=False, depth_mask=None, inverse_depth=False, lw_depth=0.,
                              use_adaptive_iters=DEFAULT_USE_ADAPTIVE_ITERS_INIT):
    os.makedirs(output_class_dir, exist_ok=True)

    mesh_index = 0
    filename, file_extension = os.path.splitext(cur_imname)
    obj_para_path = os.path.join(output_class_dir, additional_name + filename + f"_r{rank}_{class_name}" + ".pkl")

    result = "error"

    if not os.path.exists(obj_para_path) or overwrite:
        object_parameters, vis_res = find_initial_object_poses(
            instances=instances, class_name=class_name, mesh_index=mesh_index, visualize=True,
            num_iterations=num_init_iters, image=np.array(image), st=None,
            mesh_path=selected_obj_name, class_id=class_id, depth_mask=depth_mask, inverse_depth=inverse_depth,
            lw_depth=lw_depth, use_adaptive_iters=use_adaptive_iters
        )
        result = "calculated"
    else:
        info_dict = joblib.load(obj_para_path)
        object_parameters = info_dict["object_parameters"]
        result = "loaded"

    return object_parameters, obj_para_path, result


def save_init_human_object_parameters(person_parameters, object_parameters, class_name, mesh_index,
                                      selected_obj_name, img_path, obj_para_path):
    output_dict = {}
    output_dict["person_parameters"] = person_parameters
    output_dict["object_parameters"] = object_parameters
    output_dict["class_name"] = class_name
    output_dict["mesh_index"] = mesh_index
    output_dict["mesh_name"] = selected_obj_name
    output_dict["img_path"] = img_path
    joblib.dump(output_dict, obj_para_path)


def load_init_human_object_parameters(obj_para_path, pare_res_path, class_name, output_cls_dir,
                                      dataset, crop_info_path=None, image_size=DEFAULT_IMAGE_SIZE, orig_image_size=DEFAULT_IMAGE_SIZE):
    output_dict = joblib.load(obj_para_path)
    object_parameters = output_dict['object_parameters']
    selected_obj_name = output_dict['mesh_name'].replace("_f1000", "").replace("_1000", "").replace("_38995", "black")
    return object_parameters, class_name, selected_obj_name, output_dict["img_path"]


def add_extra_person_parameters(person_parameters, dataset, crop_info_path, image_size, orig_image_size, output_dict):
    crop_info = None    
    img_path = output_dict["img_path"]
    if not os.path.isfile(crop_info_path):
        if dataset == "agd20k":
            with Image.open(img_path) as img:
                crop_info = {"x1": 0, "y1": 0, "x2": img.width, "y2": img.height}
    else:
        with open(crop_info_path, "r") as f:
            crop_info = json.load(f)

    cx, cy, ww, hh = person_parameters["bboxes"][0]
    cx, cy, ww, hh = cx.item(), cy.item(), ww.item(), hh.item()
    ww -= cx
    hh -= cy
    cx, cy, ww, hh = (cx - crop_info["x1"]) * image_size / (crop_info["x2"] - crop_info["x1"]), \
                     (cy - crop_info["y1"]) * image_size / (crop_info["y2"] - crop_info["y1"]), \
                      ww * image_size / (crop_info["x2"] - crop_info["x1"]), \
                      hh * image_size / (crop_info["y2"] - crop_info["y1"])
    newbbox = np.array(
        [[cx, cy, cx + ww, cy + hh]]
    )
    person_parameters["cams_new"] = torch.from_numpy(local_to_global_cam(
        newbbox,
        person_parameters.get("pred_cam", (person_parameters.get("local_cams", torch.zeros(1))).cpu().numpy()),
        image_size,
    ).astype(np.float32))
    person_parameters["verts"] = torch.from_numpy(person_parameters.get("smpl_vertices", person_parameters.get("verts", torch.zeros(1)).cpu().numpy()).astype(np.float32))
    
    if "person_parameters" in output_dict and "masks" in output_dict["person_parameters"]:
        person_parameters["masks"] = output_dict["person_parameters"]["masks"]


