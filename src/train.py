import torch
import numpy as np
import os
from tqdm import tqdm
from zmq import device
from .visualize import visualize
from .utils import ConsoleLog
from .mask_rcnn import MaskRCNN
from .dataset import AnnotationDataset
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from . import config

console_log = ConsoleLog(lines_up_on_end=1)


def train(mask_rcnn, lr, start_epoch, epoches, unfreeze_after_epoch=None):
    mask_rcnn = MaskRCNN()
    mask_rcnn.train()
    
    opt = torch.optim.Adam(mask_rcnn.parameters(), lr=lr)
    dataset = AnnotationDataset()
    data_loader = DataLoader(dataset, shuffle=True, batch_size=1)
    for epoch in range(epoches):
        epoch = epoch + start_epoch
        if unfreeze_after_epoch is not None:
            if epoch >= unfreeze_after_epoch:
                mask_rcnn.feature_extractor.unfreeze_layers(4, 9)
                
        for batch_id, (img, boxes, cls_indexes, mask_pooling_pred_targets) in enumerate(tqdm(data_loader)):
            boxes = boxes.squeeze(0)
            cls_indexes = cls_indexes.squeeze(0)
            mask_pooling_pred_targets = mask_pooling_pred_targets.squeeze(0)
            batch_id = batch_id + 1
            
            rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss, mask_loss = mask_rcnn(
                img, 
                boxes, 
                cls_indexes,
                mask_pooling_pred_targets
            )
            
            if rpn_reg_loss is not None:
                total_loss = rpn_cls_loss + 10*rpn_reg_loss + roi_cls_loss + 10*roi_reg_loss + 1.5*mask_loss
            else:
                total_loss = rpn_cls_loss + roi_cls_loss
                
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            
            with torch.no_grad():
                console_log.print([
                    ("total_loss", total_loss.item()),
                    ("-rpn_cls_loss", rpn_cls_loss.item()),
                    ("-rpn_reg_loss", rpn_reg_loss.item() if rpn_reg_loss is not None else 0),
                    ("-roi_cls_loss", roi_cls_loss.item()),
                    ("-roi_reg_loss", roi_reg_loss.item() if rpn_reg_loss is not None else 0),
                    ("-mask_loss", mask_loss.item())
                ])
            
                if batch_id % 60 == 0:
                    visualize(mask_rcnn, f"{epoch}_batch_{batch_id}.png")
                    
        state_dict = mask_rcnn.state_dict()
        torch.save(state_dict, os.path.join(config.pths_dir, f"model_epoch_{epoch}.pth"))
        


if __name__ == "__main__":
    train()
    