import os
import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(1, project_root)

import glob
import joblib
import json
import neural_renderer as nr
import numpy as np
from psbody.mesh import Mesh
from sklearn.neighbors import KDTree
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from config.defaults import (
    ADAPTIVE_ITERS_CONVERGE_THRESH,
    ADAPTIVE_ITERS_STOPPING_TOLERANCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_JOINT_LR,
    DEFAULT_NUM_JOINT_ITERS,
    DEFAULT_USE_ADAPTIVE_ITERS,
    PERSON_LABEL_PATH,
    RENDERER,
    SMPL_FACES_PATH
)
from utils.evaluation import eval_align_both_models_from_sRt, vis_3x3_grid
from utils.geometry import center_vertices, rot6d_to_matrix
if "kaolin" in RENDERER:
    import kaolin as kal


class MLP(nn.Module):
    def forward(self, x, norm_max=-1, norm_input=None, norm_output=None):
        x_copy = x[..., :512]
        if norm_input is not None and norm_input[0] is not None:
            x = (x - norm_input[0]) / norm_input[1]
        res = self.model(x)
        if norm_max > 0:
            for bidx in range(res.shape[0]):
                if torch.norm(res[bidx, :], dim=-1) > norm_max:
                    res[bidx, :] = res[bidx, :] / (torch.norm(res[bidx, :], dim=-1)).detach() * norm_max
        if norm_output is not None and norm_output[0] is not None:
            res = res * norm_output[1] + norm_output[0] - norm_input[0]
        mapped = res + x_copy
        # print(torch.norm(mapped, dim=-1).mean().item())
        mapped = torch.nn.functional.normalize(mapped, dim=-1)
        return mapped

    def __init__(self, sizes, bias=True, act=nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
                # layers.append(torch.nn.BatchNorm1d(sizes[i + 1]))
        self.model = nn.Sequential(*layers)
        self.model[0].weight.data.normal_(0, 1e-3)
        self.model[2].weight.data.normal_(0, 1e-3)
        self.model[0].bias.data.normal_(0, 1e-3)
        self.model[2].bias.data.normal_(0, 1e-3)
        print(self.model)


# PHOSA implementation
def jointly_optimize_human_object(
    model,
    class_name,
    loss_weights=None,
    num_iterations=DEFAULT_NUM_JOINT_ITERS,
    mesh_path=None,
    contact=None,
    model_name=None,
    use_adaptive_iters=DEFAULT_USE_ADAPTIVE_ITERS,
    lr=DEFAULT_JOINT_LR,
    visualize=False,
    save_path=None,
    image_path=None,
    dataset=None,
):
    m = Mesh()
    m.load_from_file(mesh_path)
    flag = False
    num_pen_iterations = 0

    if visualize:
        seq, frame, kid = model_name.split("-")[:3]
        kid = int(kid[1])
        _, bodyCD, objCD = eval_align_both_models_from_sRt(
            model.get_verts_person()[0].detach().cpu().numpy(),
            model.int_scales_object.detach().cpu().numpy(),
            rot6d_to_matrix(model.rotations_object).detach().cpu().numpy()[0],
            model.translations_object.detach().cpu().numpy()[0],
            class_name,
            {"seq": seq, "frame": frame, "kid": kid},
            dataset,
            align="person",
            #VIS3D=True,
        )

        vis_3x3_grid(model, seq, frame, kid,
                     model.masks_object, objCD, bodyCD,
                     text_descrip="Before Joint Optimization")

    def get_save_dict():
        return {"image_path": image_path, "dataset": dataset, "class_name": class_name, "loss_weights": loss_weights, "state_dict": model.state_dict()}

    # for evaluation:
    torch.save(get_save_dict(), save_path.rsplit(".", 1)[0] + "_model_before_joint_optim.pth")

    if num_iterations > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        no_better_accumu = 0  # accumulate how many iterations have passed since last best loss
        best_loss = np.inf

        loop = tqdm(range(num_iterations))
        for num_iter in loop:
            no_better_accumu += 1
            optimizer.zero_grad()
            # {k: loss_weights[k] if "pene" not in k or _ >= 200 else 0.0 for k in loss_weights}
            loss_dict = model(loss_weights=loss_weights)
            loss_dict_weighted = {
                k: loss_dict[k] * loss_weights.get(k.replace("loss", "lw"), 1.0) for k in loss_dict if "marker" not in k
            }
            loss = sum(loss_dict_weighted.values())

            if "loss_penetration" in loss_dict.keys() and "loss_penetration_marker" not in loss_dict.keys():
                num_pen_iterations += 1

            loss_wo_clip = sum({
                k: loss_dict[k] * loss_weights.get(k.replace("loss", "lw"), 1.0) for k in loss_dict if k != "loss_clipllh" and "marker" not in k
            }.values())
            if num_iter == 0 and loss_wo_clip.data == 0. and contact is not None and loss_weights["lw_human_obj_contact"] != 0:
                flag = True
                break
            info = f'[joint] iter {num_iter}  '
            for k in loss_dict:
                if "marker" in k:
                    continue
                info += f"{k}: {loss_dict_weighted[k].data:.4f} "
            info += f" total loss {loss.data:.4f}"

            if num_iter == 0:
                loop.set_description(info.strip())

            if loss < best_loss:
                best_loss = loss
                no_better_accumu = 0
                loop.set_description(info.strip())

            loss.backward()
            optimizer.step()
        
        # to create indefinite tqdm progress bar:
        def generator():
            while True:
                yield

        if num_iter > 0 and use_adaptive_iters:
            progress = tqdm(generator())
            additional_chance = 2
            while no_better_accumu < ADAPTIVE_ITERS_STOPPING_TOLERANCE:
                num_iter += 1
                progress.update()
                no_better_accumu += 1
                optimizer.zero_grad()
                loss_dict = model(loss_weights=loss_weights)
                loss_dict_weighted = {
                    k: loss_dict[k] * loss_weights.get(k.replace("loss", "lw"), 1.0) for k in loss_dict if "marker" not in k
                }
                loss = sum(loss_dict_weighted.values())
                if "loss_penetration" in loss_dict.keys() and "loss_penetration_marker" not in loss_dict.keys():
                    num_pen_iterations += 1
                info = f'[adaptive] iter {num_iter}  '
                for k in loss_dict:
                    if "marker" in k:
                        continue
                    info += f"{k}: {loss_dict_weighted[k].data:.4f} "
                info += f" total loss: {loss.data:.4f}"
                if loss < best_loss - ADAPTIVE_ITERS_CONVERGE_THRESH:
                    best_loss = loss
                    no_better_accumu = 0
                    progress.set_description(info.strip())
                else:
                    progress.set_description((info + f'  (stall for {no_better_accumu} iters)').strip())
                loss.backward()  # loss.backward()
                optimizer.step()

                # seems like this was tried only for the human-object contact loss; let's also try it for the depth loss:
                if no_better_accumu > ADAPTIVE_ITERS_STOPPING_TOLERANCE:  # and 'loss_human_obj_contact' in loss_dict_weighted and loss_dict_weighted['loss_human_obj_contact'] > 1e4:
                    #  contact loss is still large, so we give it a second chance.
                    if additional_chance <= 0:
                        break
                    else:
                        additional_chance -= 1
                        best_loss = loss
                        no_better_accumu = 0

    if visualize:
        seq, frame, kid = model_name.split("-")[:3]
        kid = int(kid[1])
        _, bodyCD, objCD = eval_align_both_models_from_sRt(
            model.get_verts_person()[0].detach().cpu().numpy(),
            model.int_scales_object.detach().cpu().numpy(),
            rot6d_to_matrix(model.rotations_object).detach().cpu().numpy()[0],
            model.translations_object.detach().cpu().numpy()[0],
            class_name,
            dataset,
            {"seq": seq, "frame": frame, "kid": kid},
            align="person",
            #VIS3D=True,
        )
        vis_3x3_grid(model, seq, frame, kid,
                     model.masks_object, objCD, bodyCD,
                     text_descrip="After Joint Optimization")

    # for evaluation:
    torch.save(get_save_dict(), save_path.rsplit(".", 1)[0] + "_model_after_joint_optim.pth")
    return model, flag


def visualize_human_object(model, image, return_rend=False):
    # Rendered frontal image
    rend, mask = model.render(model.renderer)
    if not return_rend:
        if image.max() > 1:
            image = image / 255.0
        h, w, c = image.shape
        L = max(h, w)
        new_image = np.pad(image.copy(), ((0, L - h), (0, L - w), (0, 0)))
        new_image[mask] = rend[mask]
        new_image = (new_image[:h, :w] * 255).astype(np.uint8)

    # Rendered top-down image
    theta = 1.3
    d = 0.7
    x, y = np.cos(theta), np.sin(theta)
    mx, my, mz = model.get_verts_object().mean(dim=(0, 1)).detach().cpu().numpy()
    K = model.renderer.K
    R2 = torch.cuda.FloatTensor([[[1, 0, 0], [0, x, -y], [0, y, x]]])
    # t2 = torch.cuda.FloatTensor([mx + d - 0.3, my + d, mz])
    t2 = - R2 @ model.get_verts_object().mean(dim=(0, 1)).detach() + model.get_verts_object().mean(dim=(0, 1)).detach()
    top_renderer = nr.renderer.Renderer(
        image_size=DEFAULT_IMAGE_SIZE, K=K, R=R2, t=t2, orig_size=1
    )
    top_renderer.background_color = [1, 1, 1]
    top_renderer.light_direction = [1, 0.5, 1]
    top_renderer.light_intensity_direction = 0.3
    top_renderer.light_intensity_ambient = 0.5
    top_renderer.background_color = [1, 1, 1]
    top_down, _ = model.render(top_renderer)
    top_down = (top_down * 255).astype(np.uint8)

    # Rendered side-view image
    THETA_Y = 1.57
    xy, yy = np.cos(THETA_Y), np.sin(THETA_Y)
    Ry = torch.cuda.FloatTensor([[[xy, 0, -yy], [0, 1, 0], [yy, 0, xy]]])
    # t = torch.cuda.FloatTensor([-0.23, 0.0, 0.11])
    t = - Ry @ model.get_verts_object().mean(dim=(0, 1)).detach() + model.get_verts_object().mean(dim=(0, 1)).detach()
    K = model.renderer.K
    side_renderer = nr.renderer.Renderer(
        image_size=DEFAULT_IMAGE_SIZE, K=K, R=Ry, t=t, orig_size=1
    )
    side_renderer.background_color = [1, 1, 1]
    side_renderer.light_direction = [1, 0.5, 1]
    side_renderer.light_intensity_direction = 0.3
    side_renderer.light_intensity_ambient = 0.5
    side_renderer.background_color = [1, 1, 1]
    side_view, _ = model.render(side_renderer)
    side_view = (side_view * 255).astype(np.uint8)

    THETA_Y = 3.14
    xy, yy = np.cos(THETA_Y), np.sin(THETA_Y)
    Ry = torch.cuda.FloatTensor([[[xy, 0, -yy], [0, 1, 0], [yy, 0, xy]]])
    # t = torch.cuda.FloatTensor([-0.23, 0.0, 0.11])
    t = - Ry @ model.get_verts_object().mean(dim=(0, 1)).detach() + model.get_verts_object().mean(dim=(0, 1)).detach()
    K = model.renderer.K
    side_renderer = nr.renderer.Renderer(
        image_size=DEFAULT_IMAGE_SIZE, K=K, R=Ry, t=t, orig_size=1
    )
    side_renderer.background_color = [1, 1, 1]
    side_renderer.light_direction = [1, 0.5, 1]
    side_renderer.light_intensity_direction = 0.3
    side_renderer.light_intensity_ambient = 0.5
    side_renderer.background_color = [1, 1, 1]
    side_view2, _ = model.render(side_renderer)
    side_view2 = (side_view2 * 255).astype(np.uint8)

    THETA_Y = -1.57
    xy, yy = np.cos(THETA_Y), np.sin(THETA_Y)
    Ry = torch.cuda.FloatTensor([[[xy, 0, -yy], [0, 1, 0], [yy, 0, xy]]])
    # t = torch.cuda.FloatTensor([-0.23, 0.0, 0.11])
    t = - Ry @ model.get_verts_object().mean(dim=(0, 1)).detach() + model.get_verts_object().mean(dim=(0, 1)).detach()
    K = model.renderer.K
    side_renderer = nr.renderer.Renderer(
        image_size=DEFAULT_IMAGE_SIZE, K=K, R=Ry, t=t, orig_size=1
    )
    side_renderer.background_color = [1, 1, 1]
    side_renderer.light_direction = [1, 0.5, 1]
    side_renderer.light_intensity_direction = 0.3
    side_renderer.light_intensity_ambient = 0.5
    side_renderer.background_color = [1, 1, 1]
    side_view3, _ = model.render(side_renderer)
    side_view3 = (side_view3 * 255).astype(np.uint8)

    if return_rend:
        return (rend * 255).astype(np.uint8), top_down, side_view, side_view2, side_view3
    return new_image, top_down, side_view

