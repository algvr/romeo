import os
import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(1, project_root)

import numpy as np
from PIL import Image
import torch

from config.defaults import DEFAULT_IMAGE_SIZE
from detectron2.structures import Instances, Boxes
from utils import check_overlap, compute_iou


def load_image(img_path, remove_small_images):
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    if remove_small_images:
        print("image size: ", image.size)
        img_dim = max(w, h)
        # todo: change 300
        if img_dim < 300:
            try:
                os.remove(img_path)
            except OSError as e:
                print(e)
            return None, None

    r = min(DEFAULT_IMAGE_SIZE / w, DEFAULT_IMAGE_SIZE / h)
    w = int(r * w)
    h = int(r * h)

    image = np.array(image.resize((w, h)))
    return image, w, h, r


def segment_image(image, segmenter, detection_filter, id_obj, id_per, remove_no_detection,
                  img_path, IOU_THRESHOLD, remove_small_intersection, gt_mask_path=None, inpainted_image=None):
    instances = segmenter(image)["instances"]
    if inpainted_image is not None:
        instances_inpainted = segmenter(inpainted_image)["instances"]
        classes_ids = instances.pred_classes.cpu().detach().numpy()
        classes_ids_inpainted = instances_inpainted.pred_classes.cpu().detach().numpy()
        obj_indices = np.where(classes_ids_inpainted == id_obj)[0]
        per_indices = np.where(classes_ids == id_per)[0]

        h, w, _ = image.shape
        device = instances.pred_classes.device
        instances_new = Instances((h, w))

        boxes = instances.pred_boxes.tensor.cpu().detach().numpy()
        boxes_inpainted = instances_inpainted.pred_boxes.tensor.cpu().detach().numpy()
        instances_new.pred_boxes = Boxes(
            torch.cat((
                torch.tensor(boxes[per_indices]), 
                torch.tensor(boxes_inpainted[obj_indices]),
            ))
        )
        instances_new.scores = torch.cat((instances.scores[per_indices], instances_inpainted.scores[obj_indices]))
        instances_new.pred_masks = torch.cat((instances.pred_masks[per_indices], instances_inpainted.pred_masks[obj_indices]))
        instances_new.pred_classes = torch.cat((instances.pred_classes[per_indices], instances_inpainted.pred_classes[obj_indices]))
        instances = instances_new

    if detection_filter:
        if id_obj not in instances.pred_classes or id_per not in instances.pred_classes:
            if remove_no_detection:
                try:
                    os.remove(img_path)
                    # st.write("removed")
                except OSError as e:
                    print(e)
                return None

    if IOU_THRESHOLD != 0.0:
        classes_ids = instances.pred_classes.cpu().detach().numpy()
        obj_indices = np.where(classes_ids == id_obj)[0]
        per_indices = np.where(classes_ids == id_per)[0]
        remaining_indices = []
        boxes = instances.pred_boxes.tensor.cpu().detach().numpy()
        remove_img = True
        for o_idx in obj_indices:
            obj_box = boxes[o_idx]
            for p_idx in per_indices:
                per_box = boxes[p_idx]
                if check_overlap(obj_box, per_box):
                    iou = compute_iou(np.asarray(obj_box), np.asarray(per_box))
                    if iou >= IOU_THRESHOLD:
                        # st.write(iou)
                        remaining_indices.append(o_idx)
                        remaining_indices.append(p_idx)
                        remove_img = False
                        break
        if remove_small_intersection and remove_img:
            try:
                os.remove(img_path)
                # st.write("removed")
            except OSError as e:
                print(e)
            return None

        remaining_indices = list(set(remaining_indices))
        # print(image.shape)
        h, w, _ = image.shape
        instances_new = Instances((h, w))
        instances_new.pred_boxes = Boxes(torch.tensor(boxes[remaining_indices]))
        instances_new.scores = instances.scores[remaining_indices]
        instances_new.pred_masks = instances.pred_masks[remaining_indices]
        instances_new.pred_classes = instances.pred_classes[remaining_indices]
        instances = instances_new

    if gt_mask_path is not None:
        mask_gt = np.array(Image.open(gt_mask_path).resize((640, 480)))
        classes_ids = instances.pred_classes.cpu().detach().numpy()
        per_indices = np.where(classes_ids == id_per)[0]
        remaining_indices = per_indices
        boxes = instances.pred_boxes.tensor.cpu().detach().numpy()

        def mask2bbox(o_mask):
            return [[
                np.where(o_mask == 1)[1].min(), 
                np.where(o_mask == 1)[0].min(), 
                np.where(o_mask == 1)[1].max(), 
                np.where(o_mask == 1)[0].max(),
            ]]

        bbox_gt = mask2bbox(mask_gt)
        mask_gt = mask_gt[None, ...]

        h, w, _ = image.shape
        device = instances.pred_classes.device
        instances_new = Instances((h, w))
        instances_new.pred_boxes = Boxes(
            torch.cat((
                torch.tensor(boxes[remaining_indices]), 
                torch.tensor(bbox_gt),
            ))
        )
        instances_new.scores = torch.cat((instances.scores[remaining_indices], torch.tensor([1.], device=device)))
        instances_new.pred_masks = torch.cat((instances.pred_masks[remaining_indices], torch.tensor(mask_gt, device=device)))
        instances_new.pred_classes = torch.cat((instances.pred_classes[remaining_indices], torch.tensor([id_obj], device=device)))
        instances = instances_new

    return instances
