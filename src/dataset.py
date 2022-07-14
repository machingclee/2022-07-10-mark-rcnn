from cProfile import label
from matplotlib import image
import numpy as np
import torch
import albumentations as A
import json
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from . import config
from torchvision import transforms
from torchvision.transforms import ToPILImage
from copy import deepcopy
from typing import List, TypedDict


to_tensor = transforms.ToTensor()

torch_img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def resize_img(img):
    """
    img:  Pillow image
    """
    h, w = img.height, img.width
    if h >= w:
        ratio = config.input_height / h
        new_h, new_w = int(h * ratio), int(w * ratio)        
    else:
        ratio = config.input_width / w
        new_h, new_w = int(h * ratio), int(w * ratio)

    img = img.resize((new_w, new_h), Image.BILINEAR)
    return img, (w, h)


def pad_img(img):
    h = img.height
    w = img.width
    img = np.array(img)
    img = np.pad(img, pad_width=((0, config.input_height - h), (0, config.input_width - w), (0, 0)), mode="constant")
    img = Image.fromarray(img)
    assert img.height == config.input_height
    assert img.width == config.input_width
    return img


def resize_and_padding(img, return_window=False):
    img, (ori_w, ori_h) = resize_img(img)
    w = img.width
    h = img.height
    padding_window = (w, h)
    img = pad_img(img)

    if not return_window:
        return img
    else:
        return img, padding_window, (ori_w, ori_h)



albumentation_transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0, rotate_limit=10, p=0.7),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.5),
        A.ColorJitter(p=0.5)
    ], p=0.8),

    A.LongestMaxSize(max_size=config.input_height, interpolation=1, p=1),
    A.PadIfNeeded(
        min_height=config.input_height,
        min_width=config.input_height,
        border_mode=0,
        value=(0, 0, 0),
        position="top_left"
    ),
],
    p=1,
    bbox_params=A.BboxParams(format="pascal_voc", min_area=0.1)
)

albumentation_minimal_transform = A.Compose([
    A.LongestMaxSize(max_size=config.input_height, interpolation=1, p=1),
    A.PadIfNeeded(
        min_height=config.input_height,
        min_width=config.input_height,
        border_mode=0,
        value=(0, 0, 0),
        position="top_left"
    ),
],
    p=1, bbox_params=A.BboxParams(format="pascal_voc", min_area=0.1)
)


# def data_augmentation(img, bboxes=None, return_window=False):
#     if isinstance(img, Image.Image):
#         img = np.array(img)
#     ori_h, ori_w = img.shape[0:2]
#     if bboxes is not None:
#         transformed = albumentation_transform(image=img, bboxes=bboxes)
#         img = transformed["image"]
#         bboxes = transformed["bboxes"]
#     else:
#         transformed = albumentation_transform(image=img, bboxes=[[1, 2, 3, 4, 0]])
#         img = transformed["image"]

#     max_side = max(ori_h, ori_w)
#     ratio = config.input_width / max_side
#     # to be used by pillow, so represented in (0,0, w,h)
#     padding_window = (0, 0, ori_w * ratio, ori_h * ratio)

#     if not return_window:
#         if bboxes is not None:
#             return img, bboxes
#         else:
#             return img
#     else:
#         return img, bboxes, padding_window, (ori_w, ori_h)


def data_minimal_augmentation(img, boxes=None):
    # in order to fit into the network
    if isinstance(img, Image.Image):
        img = np.array(img)
    if boxes is None:
        boxes = [[1, 2, 3, 4, 0]]
        transformed = albumentation_minimal_transform(image=img, bboxes=boxes)
        return transformed["image"]
    else: 
        transformed = albumentation_minimal_transform(image=img, bboxes=boxes)
        return transformed["image"], transformed["bboxes"]


Segmentation = List[float]
Box = List[float]

class Target(TypedDict):
    image_path: str
    boxes: List[List[float]]
    segmentations: List[List[float]]


class AnnotationDataset(Dataset):
    def __init__(self, mode="train"):
        assert mode in ["train", "test"]
        self.mode = mode
        super(AnnotationDataset, self).__init__()
        self.annotations = {}
            
        annotation_files = config.train_annotation_files
        data = {} # {img_name: [(segmentation, cls_idx)]}
        self.data: List[Target] = []
        
        imageId_imagePath = {}
        self.cls_names = []
        
        for annotation_file in annotation_files:
            with open(annotation_file, "r") as f:
                annotations_ = json.load(f)
            images = annotations_["images"]
            
            categories = annotations_["categories"]
            annotations = annotations_["annotations"]
            
            for img in images:
                imageId_imagePath.update({img["id"]: img["file_name"]})
            for cat in categories:
                self.cls_names.append(cat["name"])
                        
            for anno in annotations:
                image_id = anno["image_id"]
                category_id = anno["category_id"]
                image_path = imageId_imagePath[image_id]
                
                bbox = anno["bbox"]
                x, y, w, h = bbox
                xmin = float(x)
                ymin = float(y)
                xmax = xmin + float(w)
                ymax = ymin + float(h)
                bbox = [xmin, ymin, xmax, ymax, category_id]
                segmentation =  anno["segmentation"]
                
                target = next((target for target in self.data if target.get("image_path")==image_path), None)
                if target is not None:
                    target["boxes"].append(bbox)
                    target["segmentations"].append(segmentation[0])
                else:
                    target: Target = {
                        "image_path": image_path,
                        "boxes": [bbox],
                        "segmentations":[segmentation[0]]                      
                    }
                    self.data.append(target)

    def __getitem__(self, index):
        data = self.data[index]
        img_path = data["image_path"]
        boxes = data["boxes"]
        boxes = np.array(boxes)
        segmentations = data["segmentations"]

        img = Image.open("data/"+img_path)
        
        mask = Image.new("L", (img.width, img.height), color="black")
        draw = ImageDraw.Draw(mask)
        
        for seg, box in zip(segmentations, boxes):
            draw.polygon(seg, fill="white", outline="white")
        img = np.array(img)
        mask = np.array(mask)

        if self.mode == "train":
            if len(boxes) > 0:
                transformed = albumentation_transform(image=img, bboxes=boxes, mask=mask)
                img = transformed["image"]
                boxes = transformed["bboxes"]
                mask = transformed["mask"]
                
                mask_pooling_pred_targets = []
                
                for box in boxes:
                    x1, y1, x2, y2, _ = box
                    cropped_mask = Image.fromarray(mask).crop((x1, y1, x2, y2))
                    cropped_mask = cropped_mask.resize((28, 28))
                    cropped_mask = np.array(cropped_mask)
                    cropped_mask = np.where(cropped_mask > 40, 1, 0)
                    mask_pooling_pred_targets.append(cropped_mask)
                
                mask_pooling_pred_targets = np.array(mask_pooling_pred_targets)
                
                img = torch_img_transform(img)
                
                if len(boxes) == 0:
                    return img, torch.as_tensor([]), torch.as_tensor([]), torch.as_tensor([])
                
                new_boxes = np.array(boxes)

                boxes_ = torch.as_tensor(new_boxes[..., 0:4]).float()
                cls_idxes = torch.as_tensor(new_boxes[..., 4]).float()
                mask_pooling_pred_targets = torch.as_tensor(mask_pooling_pred_targets).float()

                return img, boxes_, cls_idxes, mask_pooling_pred_targets
            else:
                return img, torch.as_tensor([]), torch.as_tensor([]), torch.as_tensor([])
        else:
            transformed = albumentation_minimal_transform(image=img, bboxes=boxes)
            img = transformed["image"]
            boxes = transformed["bboxes"]
            torch.as_tensor([])
            img = torch_img_transform(img)
            return img, boxes, img_path

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = AnnotationDataset()
    img, boxes_, cls_idxes, mask_pooling_pred_targets=dataset[0]
    print(img)
    print(boxes_)
    print(cls_idxes)
    print(mask_pooling_pred_targets)
