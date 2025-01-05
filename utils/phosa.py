import os
import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(1, project_root)

import itertools
import mesh_to_sdf
import neural_renderer as nr
import numpy as np
from PIL import Image
from psbody.mesh import Mesh
from scipy.ndimage.morphology import distance_transform_edt
import scipy.sparse as sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from tqdm.auto import tqdm

from config.defaults import (
    BBOX_EXPANSION,
    BBOX_EXPANSION_PARTS,
    DEFAULT_IMAGE_SIZE,
    INTERACTION_MAPPING,
    INTERACTION_THRESHOLD,
    REND_SIZE,
    DEFAULT_INTERACTION_THRESHOLD,
    RENDERER,
    USE_OPTUNA
)
from utils.bbox import check_overlap, compute_iou
from utils.camera import (
    compute_transformation_ortho,
    compute_transformation_persp,
)
import utils.chamfer_distance_torch as cd_torch_module
chamfer_distance_torch = cd_torch_module.chamferDist()
from utils.geometry import (
    combine_verts,
    compute_dist_z,
    matrix_to_rot6d,
    rot6d_to_matrix,
)
if "kaolin" in RENDERER:
    import kaolin as kal
if USE_OPTUNA:
    import optuna


def get_faces_and_textures(verts_list, faces_list):
    """
    Args:
        verts_list (List[Tensor(B x V x 3)]).
        faces_list (List[Tensor(f x 3)]).

    Returns:
        faces: (1 x F x 3)
        textures: (1 x F x 1 x 1 x 1 x 3)
    """
    colors_list = [
        [251 / 255.0, 128 / 255.0, 114 / 255.0],  # red
        [0.65098039, 0.74117647, 0.85882353],  # blue
        [0.9, 0.7, 0.7],  # pink
    ]
    all_faces_list = []
    all_textures_list = []
    o = 0
    for verts, faces, colors in zip(verts_list, faces_list, colors_list):
        B = len(verts)
        index_offset = torch.arange(B).to(verts.device) * verts.shape[1] + o
        o += verts.shape[1] * B
        faces_repeat = faces.clone().repeat(B, 1, 1)
        faces_repeat += index_offset.view(-1, 1, 1)
        faces_repeat = faces_repeat.reshape(-1, 3)
        all_faces_list.append(faces_repeat)
        textures = torch.FloatTensor(colors).to(verts.device)
        all_textures_list.append(textures.repeat(faces_repeat.shape[0], 1, 1, 1, 1))
    all_faces_list = torch.cat(all_faces_list).unsqueeze(0)
    all_textures_list = torch.cat(all_textures_list).unsqueeze(0)
    return all_faces_list, all_textures_list


def project_bbox(vertices, renderer, parts_labels=None, bbox_expansion=0.0):
    """
    Computes the 2D bounding box of the vertices after projected to the image plane.

    TODO(@jason): Batch these operations.

    Args:
        vertices (V x 3).
        renderer: Renderer used to get camera parameters.
        parts_labels (dict): Dictionary mapping a part name to the corresponding vertex
            indices.
        bbox_expansion (float): Amount to expand the bounding boxes.

    Returns:
        If a part_label dict is given, returns a dictionary mapping part name to bbox.
        Else, returns the projected 2D bounding box.
    """
    proj = nr.projection(
        (vertices * torch.tensor([[1, -1, 1.0]]).cuda()).unsqueeze(0),
        K=renderer.K,
        R=renderer.R,
        t=renderer.t,
        dist_coeffs=renderer.dist_coeffs,
        orig_size=1,
    )
    proj = proj.squeeze(0)[:, :2]
    
    if parts_labels is None or True:
        # print("Overriding object part assignment.")
        parts_labels = {"": torch.arange(len(vertices)).to(vertices.device)}

    bbox_parts = {}
    for part, inds in parts_labels.items():
        bbox = torch.cat((proj[inds].min(0).values, proj[inds].max(0).values), dim=0)
        if bbox_expansion:
            center = (bbox[:2] + bbox[2:]) / 2
            b = (bbox[2:] - bbox[:2]) / 2 * (1 + bbox_expansion)
            bbox = torch.cat((center - b, center + b))
        bbox_parts[part] = bbox
    if "" in parts_labels:
        return bbox_parts[""]
    return bbox_parts


class Losses(object):
    def __init__(
        self,
        renderer,
        ref_mask,
        keep_mask,
        person_mask,
        ratio_obj_by_person,
        depth_mask,
        K_rois,
        class_name,
        labels_person,
        labels_object,
        interaction_map_parts,
        faces_person,
        faces_object,
        compute_edges_fn=None,
        edt_ref_edge=None
    ):
        self.renderer = nr.renderer.Renderer(
            image_size=REND_SIZE, K=renderer.K, R=renderer.R, t=renderer.t, orig_size=1
        )
        self.ref_mask = ref_mask
        self.keep_mask = keep_mask
        self.person_mask = person_mask
        self.ratio_obj_by_person = ratio_obj_by_person
        self.depth_mask = depth_mask
        self.ref_depth_mask = self.depth_mask * self.ref_mask
        self.nrmzd_ref_depth_mask = self.ref_depth_mask.clone()
        B = self.nrmzd_ref_depth_mask.shape[0]
        assert B == 1
        if len(self.ref_depth_mask[self.ref_depth_mask > 0]) > 0:
            min_depth = self.ref_depth_mask[self.ref_depth_mask > 0].min()
            max_depth = self.ref_depth_mask[self.ref_depth_mask > 0].max()
            self.nrmzd_ref_depth_mask[self.nrmzd_ref_depth_mask > 0] = \
                (self.nrmzd_ref_depth_mask[self.nrmzd_ref_depth_mask > 0] - min_depth) / (max_depth - min_depth)

            # denoise the normalized depth map
            sorted_nrmzd_ref_depth_mask = torch.sort(self.nrmzd_ref_depth_mask.view(B, -1), dim=-1).values
            num_lt_0 = (sorted_nrmzd_ref_depth_mask <= 0).sum(-1)  # (B,) index of <= 0 values per batch sample
            num_gt_0 = (sorted_nrmzd_ref_depth_mask > 0).sum(-1)  # (B,) index of > 0 values per batch sample
            quantile01 = sorted_nrmzd_ref_depth_mask[torch.arange(B).unsqueeze(-1), int(num_lt_0+num_gt_0*0.01)]  # (B,1)
            quantile99 = sorted_nrmzd_ref_depth_mask[torch.arange(B).unsqueeze(-1), int(num_lt_0+num_gt_0*0.99)]  # (B,1)
            renorm01 = quantile01 > 0.2  # (B, 1) whether to re-normalization per batch sample
            self.nrmzd_ref_depth_mask[self.nrmzd_ref_depth_mask < quantile01] =\
                0 * renorm01 + self.nrmzd_ref_depth_mask[self.nrmzd_ref_depth_mask < quantile01] * (~renorm01)
            renorm99 = quantile99 < 0.8  # (B, 1) whether to re-normalization per batch sample
            self.nrmzd_ref_depth_mask[self.nrmzd_ref_depth_mask > quantile99] = \
                0 * renorm99 + self.nrmzd_ref_depth_mask[self.nrmzd_ref_depth_mask > quantile99] * (~renorm99)
            min_depth = self.nrmzd_ref_depth_mask[self.nrmzd_ref_depth_mask > 0].min()
            max_depth = self.nrmzd_ref_depth_mask[self.nrmzd_ref_depth_mask > 0].max()
            self.nrmzd_ref_depth_mask[self.nrmzd_ref_depth_mask > 0] = \
                (self.nrmzd_ref_depth_mask[self.nrmzd_ref_depth_mask > 0] - min_depth) / (max_depth - min_depth)

        self.K_rois = K_rois
        self.thresh = INTERACTION_THRESHOLD.get(class_name, DEFAULT_INTERACTION_THRESHOLD)  # z thresh for interaction loss
        self.mse = torch.nn.MSELoss()
        self.class_name = class_name
        self.labels_person = labels_person
        self.labels_object = labels_object
        self.expansion = BBOX_EXPANSION.get(class_name, 0.01)
        self.expansion_parts = BBOX_EXPANSION_PARTS.get(class_name, 0.011)
        self.interaction_map = INTERACTION_MAPPING.get(class_name, [])
        self.interaction_map_parts = interaction_map_parts
        self.interaction_pairs = None
        self.interaction_pairs_parts = None
        self.bboxes_parts_person = None
        self.bboxes_parts_object = None
        self.faces_person = faces_person
        self.faces_object = faces_object
        self.compute_edges_fn = compute_edges_fn
        self.edt_ref_edge = edt_ref_edge
        self.first_ld = None

    def assign_interaction_pairs(self, verts_person, verts_object):
        """
        Assigns pairs of people and objects that are interacting. Note that multiple
        people can be assigned to the same object, but one person cannot be assigned to
        multiple objects. (This formulation makes sense for objects like motorcycles
        and bicycles. Can be changed for handheld objects like bats or rackets).

        This is computed separately from the loss function because there are potential
        speed improvements to re-using stale interaction pairs across multiple
        iterations (although not currently being done).

        A person and an object are interacting if the 3D bounding boxes overlap:
            * Check if X-Y bounding boxes overlap by projecting to image plane (with
              some expansion defined by BBOX_EXPANSION), and
            * Check if Z overlaps by thresholding distance.

        Args:
            verts_person (N_p x V_p x 3).
            verts_object (N_o x V_o x 3).

        Returns:
            interaction_pairs: List[Tuple(person_index, object_index)]
        """
        with torch.no_grad():
            bboxes_object = [
                project_bbox(v, self.renderer, bbox_expansion=self.expansion)
                for v in verts_object
            ]
            bboxes_person = [
                project_bbox(v, self.renderer, self.labels_person, self.expansion)
                for v in verts_person
            ]
            num_people = len(bboxes_person)
            num_objects = len(bboxes_object)
            ious = np.zeros((num_people, num_objects))
            for part_person in self.interaction_map:
                for i_person in range(num_people):
                    for i_object in range(num_objects):
                        iou = compute_iou(
                            bbox1=bboxes_object[i_object],
                            bbox2=bboxes_person[i_person][part_person],
                        )
                        ious[i_person, i_object] += iou

            self.interaction_pairs = []
            for i_person in range(num_people):
                i_object = np.argmax(ious[i_person])
                if ious[i_person][i_object] == 0:
                    continue
                dist = compute_dist_z(verts_person[i_person], verts_object[i_object])
                if dist < self.thresh:
                    self.interaction_pairs.append((i_person, i_object))
            return self.interaction_pairs

    def assign_interaction_pairs_parts(self, verts_person, verts_object):
        """
        Assigns pairs of person parts and objects pairs that are interacting.

        This is computed separately from the loss function because there are potential
        speed improvements to re-using stale interaction pairs across multiple
        iterations (although not currently being done).

        A part of a person and a part of an object are interacting if the 3D bounding
        boxes overlap:
            * Check if X-Y bounding boxes overlap by projecting to image plane (with
              some expansion defined by BBOX_EXPANSION_PARTS), and
            * Check if Z overlaps by thresholding distance.

        Args:
            verts_person (N_p x V_p x 3).
            verts_object (N_o x V_o x 3).

        Returns:
            interaction_pairs_parts:
                List[Tuple(person_index, person_part, object_index, object_part)]
        """
        with torch.no_grad():
            bboxes_person = [
                project_bbox(v, self.renderer, self.labels_person, self.expansion_parts)
                for v in verts_person
            ]
            bboxes_object = [
                project_bbox(v, self.renderer, self.labels_object, self.expansion_parts)
                for v in verts_object
            ]
            self.interaction_pairs_parts = []
            self.interaction_pairs_parts_weight = []
            for i_p, i_o in itertools.product(
                range(len(verts_person)), range(len(verts_object))
            ):
                for part_object in self.interaction_map_parts.keys():
                    # object that does not contain certain parts
                    if part_object not in self.labels_object.keys():
                        continue
                    for part_person in self.interaction_map_parts[part_object]:
                        if type(part_person) == tuple:
                            part_person, part_person_weight = part_person[0], part_person[1]
                        else:
                            part_person_weight = 1.
                        
                        bbox_object = bboxes_object[i_o]  # [part_object]
                        bbox_person = bboxes_person[i_p]  # [part_person]
                        is_overlapping = check_overlap(bbox_object, bbox_person)
                        if part_person == 'head':
                            continue
                        z_dist = compute_dist_z(
                            verts_object[i_o][self.labels_object[part_object]],
                            verts_person[i_p][self.labels_person[part_person]],
                        )
                        # if is_overlapping and z_dist < self.thresh:
                        if True:  # without filtering actions
                            self.interaction_pairs_parts.append(
                                (i_p, part_person, i_o, part_object, part_person_weight)
                            )
            return self.interaction_pairs_parts

    def compute_person_depth(self, verts, faces):
        verts = verts[:1, ...]
        faces = faces[:1]
        for i in range(len(verts)):
            v = verts[i].unsqueeze(0)
            K = torch.tensor([[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]], device=v.device)
            assert 'kaolin' in RENDERER
            vertices_batch = verts
            faces_batch = faces[i]

            vertices_to_faces = nr.vertices_to_faces(vertices_batch, faces_batch)
            batch_size, num_faces = faces_batch.shape[0:2]
            R = torch.eye(3).unsqueeze(0).cuda()
            t = torch.zeros(1, 3).cuda()
            cam_rot = R.cuda().repeat(batch_size, 1, 1)
            cam_trans = t.cuda().repeat(batch_size, 1)

            face_features = torch.zeros((batch_size, num_faces, 3, 3), device=vertices_to_faces.device)
            for k in range(3):
                face_features[:, :, k, k] = 1
            faces = faces_batch[0].long()

            vertices_camera = kal.render.camera.rotate_translate_points(vertices_batch, cam_rot,
                                                                        cam_trans)

            # empirically found: we do not need to scale by the depth
            vertices_image_depth = nr.projection(vertices_batch, K, R, t, self.renderer.dist_coeffs,
                                                 1)  # keep orig_size to 1!
            vertices_image = vertices_image_depth[:, :, :2]  # / vertices_image_depth[:, :, 2:3]

            face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(vertices_camera, faces)
            face_vertices_image = kal.ops.mesh.index_vertices_by_faces(vertices_image, faces)
            rendered_features, rendered_face_idx = \
                kal.render.mesh.rasterize(self.ref_mask[0].shape[0], self.ref_mask[0].shape[0],
                                          -vertices_to_faces[:, :, :, -1],
                                          face_vertices_image, face_features)
            depth_vertices = vertices_image_depth[i, :, 2]  # (N_vertices,)
            depth_faces = depth_vertices[faces]  # (N_faces, 3)
            depth_min = torch.min(depth_faces).unsqueeze(0).unsqueeze(0).expand(-1, 3) * 0.95
            depth_faces = torch.cat((depth_faces, depth_min), 0)
            depth_rend = rendered_features[..., 0] * depth_faces[rendered_face_idx, 0] + \
                         rendered_features[..., 1] * depth_faces[rendered_face_idx, 1] + \
                         rendered_features[..., 2] * depth_faces[rendered_face_idx, 2]
            depth_rend = depth_rend * self.person_mask
        mean = depth_rend[depth_rend>0].mean().detach()
        if (depth_rend>0).sum() == 0:
            return mean, torch.zeros_like(mean)
        else:
            return mean, depth_rend[depth_rend>0].min().detach()

    def compute_sil_and_depth_loss(self, verts, faces, depth_loss=False):
        loss_sil = torch.tensor(0.0).float().cuda()
        loss_depth = torch.tensor(0.0).float().cuda()
        loss_dict = {}

        for i in range(len(verts)):
            v = verts[i].unsqueeze(0)
            K = self.K_rois[i]
            if "kaolin" in RENDERER:  # credit Alexey
                vertices_batch = verts
                faces_batch = faces[i]

                vertices_to_faces = nr.vertices_to_faces(vertices_batch, faces_batch)
                batch_size, num_faces = faces_batch.shape[0:2]
                R = torch.eye(3).unsqueeze(0).cuda()
                t = torch.zeros(1, 3).cuda()
                cam_rot = R.cuda().repeat(batch_size, 1, 1)
                cam_trans = t.cuda().repeat(batch_size, 1)

                face_features = torch.zeros((batch_size, num_faces, 3, 3), device=vertices_to_faces.device)
                for k in range(3):
                    face_features[:, :, k, k] = 1
                faces = faces_batch[0].long()

                vertices_camera = kal.render.camera.rotate_translate_points(vertices_batch, cam_rot,
                                                                            cam_trans)

                # empirically found: we do not need to scale by the depth
                vertices_image_depth = nr.projection(vertices_batch, K, R, t, self.renderer.dist_coeffs,
                                                     1)  # keep orig_size to 1!
                vertices_image = vertices_image_depth[:, :, :2]  # / vertices_image_depth[:, :, 2:3]

                face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(vertices_camera, faces)
                face_vertices_image = kal.ops.mesh.index_vertices_by_faces(vertices_image, faces)
                face_normals = kal.ops.mesh.face_normals(face_vertices_camera, unit=True)

                rendered_features, rendered_face_idx = \
                    kal.render.mesh.rasterize(self.ref_mask[0].shape[0], self.ref_mask[0].shape[0], -vertices_to_faces[:, :, :, -1],
                                              face_vertices_image, face_features)
                rend = kal.render.mesh.dibr_soft_mask(face_vertices_image, rendered_face_idx, sigmainv=1e7)

                depth_loss = depth_loss and not torch.isnan(self.ratio_obj_by_person) and self.avg_person_depth > 0
                if depth_loss:
                    depth_vertices = vertices_image_depth[i, :, 2]  # (N_vertices,)
                    depth_faces = torch.gather(depth_vertices.unsqueeze(-1).expand(-1, 3), 0, faces)
                    depth_min = torch.min(depth_faces).unsqueeze(0).unsqueeze(0).expand(-1, 3) * 0.95
                    depth_faces = torch.cat((depth_faces, depth_min), 0)
                    rendered_face_idx2 = rendered_face_idx.masked_fill(rendered_face_idx == -1, faces.shape[0])
                    depth_rend = rendered_features[..., 0] * torch.gather(depth_faces[:, 0].unsqueeze(0).unsqueeze(-1).expand(-1,-1,rendered_face_idx2.shape[-1]), 1, rendered_face_idx2)+\
                                 rendered_features[..., 1] * torch.gather(depth_faces[:, 1].unsqueeze(0).unsqueeze(-1).expand(-1,-1,rendered_face_idx2.shape[-1]), 1, rendered_face_idx2)+\
                                 rendered_features[..., 2] * torch.gather(depth_faces[:, 2].unsqueeze(0).unsqueeze(-1).expand(-1,-1,rendered_face_idx2.shape[-1]), 1, rendered_face_idx2)
                    
                    occlu_aware_depth_rend = depth_rend * self.keep_mask
                    avg_object_depth = occlu_aware_depth_rend[occlu_aware_depth_rend>0].mean()
                    min_object_depth = occlu_aware_depth_rend[occlu_aware_depth_rend>0].min().detach()
            else:
                rend = self.renderer(v, faces[i], K=K, mode="silhouettes")
                if depth_loss:
                    assert False, 'depth loss is only supported when kaolin=True'

            image = self.keep_mask[i] * rend
            l_m = ((image - self.ref_mask[i]) ** 2.0).sum() / self.keep_mask[i].sum()
            loss_sil += l_m
            if depth_loss:
                l_d = (  (avg_object_depth / self.avg_person_depth - self.ratio_obj_by_person) / self.ratio_obj_by_person  ) ** 2.0
                loss_depth += l_d

        if self.first_ld is None:
            self.first_ld = loss_depth.detach() / (loss_sil.detach())

        if not depth_loss:
            return {**loss_dict, "loss_sil": loss_sil / len(verts)}
        else:
            return {**loss_dict, "loss_sil": loss_sil / len(verts), "loss_reldepth": loss_depth / (self.first_ld * len(verts))}

    def compute_interaction_loss(self, verts_person, verts_object):
        """
        Computes interaction loss.
        """
        loss_interaction = torch.tensor(0.0).float().cuda()
        interaction_pairs = self.assign_interaction_pairs(verts_person, verts_object)
        for i_person, i_object in interaction_pairs:
            v_person = verts_person[i_person]
            v_object = verts_object[i_object]
            centroid_error = self.mse(v_person.mean(0), v_object.mean(0))
            loss_interaction += centroid_error
        num_interactions = max(len(interaction_pairs), 1)
        return {"loss_inter": loss_interaction / num_interactions}

    def compute_interaction_loss_parts(self, verts_person, verts_object):
        loss_interaction_parts = torch.tensor(0.0).float().cuda()
        interaction_pairs_parts = self.assign_interaction_pairs_parts(
            verts_person=verts_person, verts_object=verts_object
        )
        for i_p, part_p, i_o, part_o, part_person_weight in interaction_pairs_parts:
            v_person = verts_person[i_p][self.labels_person[part_p]]
            v_object = verts_object[i_o][self.labels_object[part_o]]
            dist = self.mse(v_person.mean(0), v_object.mean(0))
            loss_interaction_parts += dist * part_person_weight
        num_interactions = max(len(self.interaction_pairs_parts), 1)
        return {"loss_inter_part": loss_interaction_parts / num_interactions}

    def compute_intrinsic_scale_prior(self, intrinsic_scales, intrinsic_mean):
        return torch.sum((intrinsic_scales - intrinsic_mean) ** 2)

    def compute_ordinal_depth_loss(self, masks, silhouettes, depths):
        loss = torch.tensor(0.0).float().cuda()
        num_pairs = 0
        for i in range(len(silhouettes)):
            for j in range(len(silhouettes)):
                has_pred = silhouettes[i] & silhouettes[j]
                if has_pred.sum() == 0:
                    continue
                else:
                    num_pairs += 1
                front_i_gt = masks[i] & (~masks[j])
                front_j_pred = depths[j] < depths[i]
                m = front_i_gt & front_j_pred & has_pred
                if m.sum() == 0:
                    continue
                dists = torch.clamp(depths[i] - depths[j], min=0.0, max=2.0)
                loss += torch.sum(torch.log(1 + torch.exp(dists))[m])
        loss /= num_pairs
        return {"loss_depth": loss}

    def compute_common_ground_contact_loss(self, verts_person, verts_object, obj_base_idx):
        loss_ground_contact = torch.tensor(0.0).float().cuda()
        max_y = torch.max(verts_person[0, :, 1])
        obj_base_vert = verts_object[0][obj_base_idx]
        # same height constraint
        loss_ground_contact = self.mse(obj_base_vert[:, 1], max_y)
        return {"loss_ground_contact": loss_ground_contact}
    
    def compute_human_obj_contact_loss(self, verts_person, verts_object):
        loss = torch.tensor(0.0).float().cuda()
        interaction_pairs_parts = self.assign_interaction_pairs_parts(
            verts_person=verts_person, verts_object=verts_object
        )
        for i_p, part_p, i_o, part_o, part_person_weight in interaction_pairs_parts:
            # only for chair now:
            if not part_o == 'Chair Seat' and 'Chair' in part_o:
                seat_center = verts_object[i_o][self.labels_object['Chair Seat']].mean(0)

            v_person = verts_person[i_p][self.labels_person[part_p]]
            v_object = verts_object[i_o][self.labels_object[part_o]]
            
            # contact normal:
            person_mesh = Mesh(v=verts_person[i_p].detach().cpu().numpy(), f=self.faces_person.squeeze().detach().cpu().numpy())
            body_triangles = torch.index_select(verts_person[i_p].unsqueeze(0), 1, self.faces_person.to(torch.int64).flatten()).view(1, -1, 3, 3)
            edge0 = body_triangles[:, :, 1] - body_triangles[:, :, 0]
            edge1 = body_triangles[:, :, 2] - body_triangles[:, :, 0]
            body_normals = torch.cross(edge0, edge1, dim=2)
            body_normals = body_normals / torch.norm(body_normals, 2, dim=2, keepdim=True)
            # compute the vertex normals
            ftov = person_mesh.faces_by_vertex(as_sparse_matrix=True)
            ftov = sparse.coo_matrix(ftov)
            indices = torch.LongTensor(np.vstack((ftov.row, ftov.col))).cuda()
            values = torch.FloatTensor(ftov.data).cuda()
            ftov = torch.sparse.FloatTensor(indices, values, torch.Size(ftov.shape))
            body_v_normals = torch.mm(ftov, body_normals.squeeze())
            body_v_normals = body_v_normals / torch.norm(body_v_normals, 2, dim=1, keepdim=True)
            # vertex normals of contact vertices
            contact_body_verts_normals = body_v_normals[self.labels_person[part_p], :]
            
            object_mesh = Mesh(v=verts_object[i_o].detach().cpu().numpy(), f=self.faces_object.squeeze().detach().cpu().numpy())
            object_vn = torch.tensor(object_mesh.estimate_vertex_normals()).float().cuda() 
            # get inner surface
            if part_o == 'Chair Seat' or 'Chair' not in part_o:
                inner_idx = F.cosine_similarity(torch.tensor([[0.,-1.,0.]]).cuda(), object_vn[self.labels_object[part_o]]) > 0
                # v_object_inner_id = self.labels_object[part_o][torch.where(inner_idx)[0]]
                # v_object = verts_object[:, v_object_inner_id, :]
            else:
                dir_to_center = v_object - seat_center
                inner_idx = F.cosine_similarity(dir_to_center, object_vn[self.labels_object[part_o]]) < 0
            v_object_inner_id = self.labels_object[part_o][torch.where(inner_idx)[0]]
            v_object = verts_object[i_o, v_object_inner_id, :].squeeze()
            contact_object_verts_normals = object_vn[v_object_inner_id, :]

            contact_dist, _, idx1, _ = chamfer_distance_torch(v_person.unsqueeze(0).contiguous(), v_object.unsqueeze(0).contiguous())
            loss += contact_dist.mean() * part_person_weight

            # contact_object_verts_normals = object_vn[idx1.squeeze().long(), :]
            loss += 0.01 * (1 + F.cosine_similarity(contact_body_verts_normals, contact_object_verts_normals[idx1.squeeze().long()]).mean()) / 2 * part_person_weight
        num_interactions = max(len(self.interaction_pairs_parts), 1)
        return {"loss_human_obj_contact": loss / num_interactions}

    def get_sdf(self, verts, faces, grid_dim, vmin, vmax):
        mesh = trimesh.Trimesh(verts, faces, process=False)
        d1 = torch.linspace(vmin[0], vmax[0], grid_dim)
        d2 = torch.linspace(vmin[1], vmax[1], grid_dim)
        d3 = torch.linspace(vmin[2], vmax[2], grid_dim)
        meshx, meshy, meshz = torch.meshgrid((d1, d2, d3))
        qp = {
                (i,j,h): (meshx[i,j,h].item(), meshy[i,j,h].item(), meshz[i,j,h].item()) 
                for (i,j,h) in itertools.product(range(grid_dim), range(grid_dim), range(grid_dim))
        }
        qp_idxs = list(qp.keys())
        qp_values = np.array(list(qp.values()))
        qp_sdfs = mesh_to_sdf.mesh_to_sdf(mesh, qp_values)
        qp_map = {qp_idxs[k]: qp_sdfs[k] for k in range(len(qp_sdfs))}
        qp_sdfs = np.zeros((grid_dim, grid_dim, grid_dim))
        for (i,j,h) in itertools.product(range(grid_dim), range(grid_dim), range(grid_dim)):
            qp_sdfs[i,j,h] = qp_map[(i,j,h)]
        qp_sdfs = torch.tensor(qp_sdfs)
        return qp_sdfs

    def compute_penetration_loss(self, verts_person, faces_person, verts_object):
        loss = torch.tensor(0.0).float().cuda()
        interaction_pairs_parts = self.assign_interaction_pairs_parts(
            verts_person=verts_person, verts_object=verts_object
        ) if self.interaction_map_parts is not None else self.assign_interaction_pairs(verts_person, verts_object)

        with torch.no_grad():
            person_bbox_min, _ = torch.min(verts_person, axis=-2)
            person_bbox_max, _ = torch.max(verts_person, axis=-2)

            object_bbox_min, _ = torch.min(verts_object, axis=-2)
            object_bbox_max, _ = torch.max(verts_object, axis=-2)

            intersect_min = torch.max(person_bbox_min, object_bbox_min)
            intersect_max = torch.min(person_bbox_max, object_bbox_max)

            if not torch.all(intersect_min < intersect_max):
                return {}

        faces_person = faces_person.detach().cpu().numpy()
        p_o_pairs = set()
        for _ in interaction_pairs_parts:
            try:
                i_p, part_p, i_o, part_o, part_p_weight = _
            except:
                i_p, i_o = _
            p_o_pairs.add((i_p, i_o))

        intersect_min_np = intersect_min.squeeze().detach().cpu().numpy()
        intersect_max_np = intersect_max.squeeze().detach().cpu().numpy()



        for p_o_pair in p_o_pairs:
            i_p, i_o = p_o_pair
            v_person = verts_person[i_p].detach().cpu().numpy()
            v_object = verts_object[i_o]
            vmin = v_person.min(0) 
            vmax = v_person.max(0)
            vmin_tensor = torch.tensor(vmin).cuda()
            vmax_tensor = torch.tensor(vmax).cuda()

            v_person_selection_mask = np.all(np.logical_and(intersect_min_np < v_person, v_person < intersect_max_np), axis=-1)
            v_person_subsample_idxs = np.where(v_person_selection_mask)[0]

            v_person_subsample = v_person[ v_person_selection_mask ]

            face_selection_mask = np.in1d(faces_person.flatten(), v_person_subsample_idxs).reshape(-1, faces_person.shape[-1]).all(axis=-1)

            if not np.any(face_selection_mask):
                return {"loss_penetration": loss, "loss_penetration_marker": 0.0}

            return {"loss_penetration": loss}
                        
            vtx_remap = np.zeros(v_person.shape[0])
            vtx_remap[v_person_subsample_idxs] = np.arange(len(v_person_subsample_idxs)).astype(int)

            f_person_subsample = faces_person[ face_selection_mask ]

            f_person_subsample_remapped = vtx_remap[f_person_subsample.flatten()].reshape(-1, faces_person.shape[-1])

            v_object_selection_mask = torch.all(torch.logical_and(intersect_min.squeeze() < v_object, v_object < intersect_max.squeeze()), dim=-1)

            v_object_subsample = v_object[ v_object_selection_mask ]

            vmin_subsample = v_person_subsample.min(0) 
            vmax_subsample = v_person_subsample.max(0)
            vmin_tensor_subsample = torch.tensor(vmin_subsample).cuda()
            vmax_tensor_subsample = torch.tensor(vmax_subsample).cuda()
            
            person_area = (vmax[0] - vmin[0]) * (vmax[1] - vmin[1]) * (vmax[2] - vmin[2])
            person_area_subsample = (vmax_subsample[0] - vmin_subsample[0]) * (vmax_subsample[1] - vmin_subsample[1]) * (vmax_subsample[2] - vmin_subsample[2])
            grid_dim = 8 # int(max(3, np.ceil(8 * person_area_subsample / person_area))) # hyperparam

            qp_sdfs = self.get_sdf(v_person_subsample, f_person_subsample_remapped, grid_dim, vmin_subsample, vmax_subsample).cuda()

            norm_verts = (v_object_subsample-vmin_tensor_subsample)/(vmax_tensor-vmin_tensor_subsample)*2-1
            item_sdf = F.grid_sample(
                        qp_sdfs.reshape(1, 1, grid_dim, grid_dim, grid_dim).float(),
                        norm_verts.reshape(1, -1, 1, 1, 3), padding_mode='border', align_corners=True)
            loss += (item_sdf[item_sdf < 0].unsqueeze(dim=-1).abs()).pow(2).sum(dim=-1).sqrt().sum()
        return {"loss_penetration": loss / max(len(p_o_pairs), 1)}

    @staticmethod
    def _compute_iou_1d(a, b):
        """
        a: (2).
        b: (2).
        """
        o_l = torch.min(a[0], b[0])
        o_r = torch.max(a[1], b[1])
        i_l = torch.max(a[0], b[0])
        i_r = torch.min(a[1], b[1])
        inter = torch.clamp(i_r - i_l, min=0)
        return inter / (o_r - o_l)


class PHOSA(nn.Module):
    def __init__(
        self,
        translations_object,
        rotations_object,
        verts_object_og,
        faces_object,
        cams_person,
        verts_person_og,
        faces_person,
        person_mask,  # used for identifying the area of person depth
        ratio_obj_by_person,
        masks_object,
        masks_person,
        K_rois,
        target_masks,
        depth_mask,
        labels_person,
        labels_object,
        interaction_map_parts,
        class_name,
        int_scale_init=1.0,
        # loss_weights=None
        cropped_image=None,
    ):
        super(PHOSA, self).__init__()
        translation_init = translations_object.detach().clone()
        self.translations_object = nn.Parameter(translation_init, requires_grad=True)
        rotations_object = rotations_object.detach().clone()
        if rotations_object.shape[-1] == 3:
            rotations_object = matrix_to_rot6d(rotations_object)
        self.rotations_object = nn.Parameter(rotations_object, requires_grad=True)

        self.register_buffer("verts_object_og", verts_object_og)
        self.register_buffer("cams_person", cams_person)
        self.register_buffer("verts_person_og", verts_person_og)
        self.register_buffer("person_mask", person_mask)
        self.register_buffer("ratio_obj_by_person", torch.tensor(ratio_obj_by_person))

        kernel_size = 7
        self.pool = torch.nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=(kernel_size // 2)
        )

        self.int_scales_object = nn.Parameter(
            int_scale_init * torch.ones(rotations_object.shape[0]).float(),
            requires_grad=False,
        )
        self.int_scale_object_mean = nn.Parameter(
            torch.tensor(int_scale_init).float(), requires_grad=False
        )
        self.int_scales_person = nn.Parameter(
            torch.ones(cams_person.shape[0]).float(), requires_grad=False
        )
        self.int_scale_person_mean = nn.Parameter(
            torch.tensor(1.0).float().cuda(), requires_grad=False
        )
        
        mask_edge = self.compute_edges(target_masks).cpu().numpy()
        power = 0.25
        edt = distance_transform_edt(1 - (mask_edge > 0)) ** (power * 2)
        self.register_buffer(
            "edt_ref_edge", torch.from_numpy(edt).float()
        )
        # self.object_base_vert_idx = self.find_object_ground_vert(verts_object_og, labels_object)

        self.register_buffer("ref_mask", (target_masks > 0).float())
        self.register_buffer("keep_mask", (target_masks >= 0).float())
        self.register_buffer("depth_mask", depth_mask)
        self.register_buffer("K_rois", K_rois)
        self.register_buffer("faces_object", faces_object.unsqueeze(0))
        self.register_buffer(
            "textures_object", torch.ones(1, len(faces_object), 1, 1, 1, 3)
        )
        self.register_buffer(
            "textures_person", torch.ones(1, len(faces_person), 1, 1, 1, 3)
        )
        self.register_buffer("faces_person", faces_person.unsqueeze(0))
        self.cuda()

        # Setup renderer
        K = torch.cuda.FloatTensor([[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]])
        R = torch.cuda.FloatTensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        t = torch.zeros(1, 3).cuda()
        self.renderer = nr.renderer.Renderer(
            image_size=DEFAULT_IMAGE_SIZE, K=K, R=R, t=t, orig_size=1
        )
        self.renderer.light_direction = [1, 0.5, 1]
        self.renderer.light_intensity_direction = 0.3
        self.renderer.light_intensity_ambient = 0.5
        self.renderer.background_color = [1, 1, 1]

        self.renderer_clip = nr.renderer.Renderer(
            image_size=224, K=K, R=R, t=t, orig_size=1
        )
        self.renderer_clip.light_direction = [1, 0.5, 1]
        self.renderer_clip.light_intensity_direction = 0.3
        self.renderer_clip.light_intensity_ambient = 0.5
        self.renderer_clip.background_color = [1, 1, 1]

        self.side_renderers_clip = []
        for THETA_Y in [-1.57, 1.57, 3.14]:
            xy, yy = np.cos(THETA_Y), np.sin(THETA_Y)
            Ry = torch.cuda.FloatTensor([[[xy, 0, -yy], [0, 1, 0], [yy, 0, xy]]])
            t = - Ry @ self.get_verts_person().mean(dim=(0, 1)).detach() + self.get_verts_person().mean(dim=(0, 1)).detach()
            K = self.renderer.K
            side_renderer = nr.renderer.Renderer(
                image_size=224, K=K, R=Ry, t=t, orig_size=1
            )
            side_renderer.background_color = [1, 1, 1]
            side_renderer.light_direction = [1, 0.5, 1]
            side_renderer.light_intensity_direction = 0.3
            side_renderer.light_intensity_ambient = 0.5
            side_renderer.background_color = [1, 1, 1]
            self.side_renderers_clip.append(side_renderer)

        """
        # random camera
        for THETA_Y in np.linspace(0, 2 * np.pi, 7)[1:-1]:
            xy, yy = np.cos(THETA_Y), np.sin(THETA_Y)
            Ry = torch.cuda.FloatTensor([[[xy, 0, -yy], [0, 1, 0], [yy, 0, xy]]])
            t = - Ry @ self.get_verts_person().mean(dim=(0, 1)).detach() + self.get_verts_person().mean(dim=(0, 1)).detach()
            K = self.renderer.K
            side_renderer = nr.renderer.Renderer(
                image_size=224, K=K, R=Ry, t=t, orig_size=1
            )
            side_renderer.background_color = [1, 1, 1]
            side_renderer.light_direction = [1, 0.5, 1]
            side_renderer.light_intensity_direction = 0.3
            side_renderer.light_intensity_ambient = 0.5
            side_renderer.background_color = [1, 1, 1]
            self.side_renderers_clip.append(side_renderer)
        """

        self.img_cropped = cropped_image

        self.register_buffer("masks_human", masks_person)
        self.register_buffer("masks_object", masks_object)
        verts_object = self.get_verts_object()
        verts_person = self.get_verts_person()
        faces, textures = get_faces_and_textures(
            [verts_object, verts_person], [faces_object, faces_person]
        )
        self.register_buffer('faces', faces)  # to store them in model's state_dict
        self.register_buffer('textures', textures)  # to store them in model's state_dict

        self.losses = Losses(
            renderer=self.renderer,
            ref_mask=self.ref_mask,
            keep_mask=self.keep_mask,
            depth_mask=self.depth_mask,
            person_mask=self.person_mask,
            ratio_obj_by_person=self.ratio_obj_by_person,
            K_rois=self.K_rois,
            interaction_map_parts=interaction_map_parts,
            labels_person=labels_person,
            labels_object=labels_object,
            class_name=class_name,
            faces_person=self.faces_person,
            faces_object=self.faces_object,
            compute_edges_fn=self.compute_edges,
            edt_ref_edge=self.edt_ref_edge
        )
        if self.person_mask is not None:
            self.losses.avg_person_depth, self.losses.min_person_depth = self.losses.compute_person_depth(verts=verts_person,
                                                                        faces=[self.faces_person] * len(verts_person))
        self.interaction_map_parts = interaction_map_parts
        
    def compute_edges(self, silhouette):
        return self.pool(silhouette) - silhouette

    def find_object_ground_vert(self, verts, labels):
        # TODO: determine the coordinate system
        chair_base_inx = labels["Chair Base"]
        obj_base_verts = verts[chair_base_inx]
        # min_y = torch.min(obj_base_verts[:, 1])
        # max_y = torch.max(verts[:, 1])
        max_y = torch.max(obj_base_verts[:, 1])
        # delta = (max_y - min_y) * 0.02
        # indices = torch.where(obj_base_verts[:, 1] < min_y + delta)[0]
        indices = torch.where(obj_base_verts[:, 1] >= max_y * 0.995)[0]
        res = chair_base_inx[indices]
        return res

    def assign_human_masks(self, masks_human=None, min_overlap=0.5):
        """
        Uses a greedy matching algorithm to assign masks to human instances. The
        assigned human masks are used to compute the ordinal depth loss.

        If the human predictor uses the same instances as the segmentation algorithm,
        then this greedy assignment is unnecessary as the human instances will already
        have corresponding masks.

        1. Compute IOU between all human silhouettes and human masks
        2. Sort IOUs
        3. Assign people to masks in order, skipping people and masks that
            have already been assigned.

        Args:
            masks_human: Human bitmask tensor from instance segmentation algorithm.
            min_overlap (float): Minimum IOU threshold to assign the human mask to a
                human instance.

        Returns:
            N_h x
        """
        f = self.faces_person
        verts_person = self.get_verts_person()
        if masks_human is None:
            return torch.zeros(verts_person.shape[0], DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE).cuda()
        person_silhouettes = torch.cat(
            [self.renderer(v.unsqueeze(0), f, mode="silhouettes") for v in verts_person]
        ).bool()

        intersection = masks_human.unsqueeze(0) & person_silhouettes.unsqueeze(1)
        union = masks_human.unsqueeze(0) | person_silhouettes.unsqueeze(1)

        iou = intersection.sum(dim=(2, 3)).float() / union.sum(dim=(2, 3)).float()
        iou = iou.cpu().numpy()
        # https://stackoverflow.com/questions/30577375
        best_indices = np.dstack(np.unravel_index(np.argsort(-iou.ravel()), iou.shape))[
            0
        ]
        human_indices_used = set()
        mask_indices_used = set()
        # If no match found, mask will just be empty, incurring 0 loss for depth.
        human_masks = torch.zeros(verts_person.shape[0], DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE).bool()
        for human_index, mask_index in best_indices:
            if human_index in human_indices_used:
                continue
            if mask_index in mask_indices_used:
                continue
            if iou[human_index, mask_index] < min_overlap:
                break
            human_masks[human_index] = masks_human[mask_index]
            human_indices_used.add(human_index)
            mask_indices_used.add(mask_index)
        return human_masks.cuda()

    def get_verts_object(self):
        return compute_transformation_persp(
            meshes=self.verts_object_og,
            translations=self.translations_object,
            rotations=rot6d_to_matrix(self.rotations_object),
            intrinsic_scales=self.int_scales_object,
        )

    def get_verts_person(self):
        return compute_transformation_ortho(
            meshes=self.verts_person_og,
            cams=self.cams_person,
            intrinsic_scales=self.int_scales_person,
            focal_length=1.0,
        )

    def compute_ordinal_depth_loss(self):
        verts_object = self.get_verts_object()
        verts_person = self.get_verts_person()

        silhouettes = []
        depths = []

        for v in verts_object:
            _, depth, sil = self.renderer.render(
                v.unsqueeze(0), self.faces_object, self.textures_object
            )
            depths.append(depth)
            silhouettes.append((sil == 1).bool())
        for v in verts_person:
            _, depth, sil = self.renderer.render(
                v.unsqueeze(0), self.faces_person, self.textures_person
            )
            depths.append(depth)
            silhouettes.append((sil == 1).bool())
        masks = torch.cat((self.masks_object, self.masks_human))
        return self.losses.compute_ordinal_depth_loss(masks, silhouettes, depths)

    def forward(self, loss_weights=None):
        """
        If a loss weight is zero, that loss isn't computed (to avoid unnecessary
        compute).
        """
        if loss_weights.get("lw_scale", 0.0) > 0.0:
            self.self.int_scales_object.requires_grad_(True)

        if loss_weights.get("lw_scale_person", 0.0) > 0.0:
            self.self.int_scales_person.requires_grad_(True)

        loss_dict = {}
        verts_object = self.get_verts_object()
        verts_person = self.get_verts_person()

        if loss_weights is None or loss_weights["lw_sil"] > 0:
            depth_loss = True if 'lw_reldepth' in loss_weights and loss_weights['lw_reldepth'] > 0 else False
            loss_dict.update(
                self.losses.compute_sil_and_depth_loss(
                    verts=verts_object, faces=[self.faces_object] * len(verts_object), depth_loss=depth_loss,
                )
            )
        if loss_weights is None or loss_weights["lw_inter"] > 0:
            loss_dict.update(
                self.losses.compute_interaction_loss(
                    verts_person=verts_person, verts_object=verts_object
                )
            )
        if loss_weights is None or loss_weights["lw_inter_part"] > 0:
            loss_dict.update(
                self.losses.compute_interaction_loss_parts(
                    verts_person=verts_person, verts_object=verts_object
                )
            )
        if loss_weights is None or loss_weights["lw_scale"] > 0:
            loss_dict["loss_scale"] = self.losses.compute_intrinsic_scale_prior(
                intrinsic_scales=self.int_scales_object,
                intrinsic_mean=self.int_scale_object_mean,
            )
        if loss_weights is None or loss_weights["lw_scale_person"] > 0:
            loss_dict["loss_scale_person"] = self.losses.compute_intrinsic_scale_prior(
                intrinsic_scales=self.int_scales_person,
                intrinsic_mean=self.int_scale_person_mean,
            )
        if loss_weights is None or loss_weights["lw_depth"] > 0:
            loss_dict.update(self.compute_ordinal_depth_loss())

        if loss_weights is None or loss_weights['lw_ground_contact'] > 0:
            loss_dict.update(
                self.losses.compute_common_ground_contact_loss(
                    verts_person=verts_person, verts_object=verts_object,
                    obj_base_idx=self.object_base_vert_idx
                )
            )
        if loss_weights is None or loss_weights['lw_human_obj_contact'] > 0:
            if self.interaction_map_parts is None:
                loss_dict.update(self.losses.compute_interaction_loss(verts_person, verts_object))
            else:
                loss_dict.update(self.losses.compute_human_obj_contact_loss(verts_person, verts_object))

        if loss_weights is None or loss_weights['lw_penetration'] > 0:
            loss_dict.update(self.losses.compute_penetration_loss(verts_person, self.faces_person.squeeze(0), verts_object))

        return loss_dict

    def render(self, renderer):
        verts_combined = combine_verts(
            [self.get_verts_object(), self.get_verts_person()]
        )
        image, _, mask = renderer.render(
            vertices=verts_combined, faces=self.faces, textures=self.textures
        )
        image = np.clip(image[0].detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
        mask = mask[0].detach().cpu().numpy().astype(bool)
        return image, mask

    def render_tensor(self, renderer):
        verts_combined = combine_verts(
            [self.get_verts_object(), self.get_verts_person()]
        )
        image, _, mask = renderer.render(
            vertices=verts_combined, faces=self.faces, textures=self.textures
        )
        image = torch.clip(image[0], 0, 1)
        mask = mask[0]


        return image, mask

    def get_parameters(self):
        """
        Computes a json-serializable dictionary of optimized parameters.

        Returns:
            parameters (dict): Dictionary mapping parameter name to list.
        """
        parameters = {
            "scales_object": self.int_scales_object,
            "scales_person": self.int_scales_person,
            "rotations_object": rot6d_to_matrix(self.rotations_object),
            "translations_object": self.translations_object,
        }
        for k, v in parameters.items():
            parameters[k] = v.detach().cpu().numpy().tolist()
        return parameters

    def save_obj(self, fname):
        with open(fname, "w") as fp:
            verts_combined = combine_verts(
                [self.get_verts_object(), self.get_verts_person()]
            )
            for v in tqdm(verts_combined[0]):
                fp.write(f"v {v[0]:f} {v[1]:f} {v[2]:f}\n")
            o = 1
            for f in tqdm(self.faces[0]):
                fp.write(f"f {f[0] + o:d} {f[1] + o:d} {f[2] + o:d}\n")
