from genericpath import exists
from sys import flags
import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.mse = nn.MSELoss(reduction="sum")
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
    
    def forward(self, predictions, target):
        #(N, 1470) -> (N, S, S, 30)
        predictions = torch.reshape(predictions, (-1, self.S, self.S, self.C+self.B*5))
        
        iou_b1 = intersection_over_union(predictions[:,:,:, 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        #unsqueezing is like wrapping each tensor in a list so we can stack them
        
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim = 0)
        iou_max , best_box = torch.max(ious,dim = 0) #best_box contains the index 
        exists_box = target[..., 20].unsqueeze(3) # Iobj_i.... prob that box exists
        
        
        # ==================== #
        # Box Coordinates #
        # ==================== #
        box_predictions = exists_box* (
            best_box*predictions[..., 26:30]
            + (1-best_box)*predictions[..., 21:25]
        )
        box_targets = exists_box* target[..., 21:25]
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4])*torch.sqrt(torch.abs(predictions[...,2:4]) + 1e-16)
        
        box_targets  = torch.sqrt(target[..., 2:4])
        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim= -2)
        )
        
        
        # ==================== #
        # For Object Loss #
        # ==================== #
        
        pred_box = exists_box*(
            best_box*predictions[...,20] + (1-best_box) * predictions[..., 25]
        )
        target_box = exists_box*target[..., 20]
        pred_loss = self.mse(
            torch.flatten(pred_box, end_dim=-2),
            torch.flatten(target_box, end_dim=-2)
            )
        
        
        
        # ==================== #
        # For No Object Loss #
        # ==================== #
        no_obj_loss = self.mse(
            torch.flatten(
                (1-exists_box)*predictions[..., 20:21], start_dim = 1
            ),
            torch.flatten(
                (1-exists_box)*target_box[..., 20:21], start_dim = 1
            )
        )
        no_obj_loss = self.mse(
            torch.flatten(
                (1-exists_box)*predictions[..., 25:26], start_dim = 1
            ),
            torch.flatten(
                (1-exists_box)*target_box[..., 20:21], start_dim = 1
            )
        )
        
        # ==================== #
        # For Class Loss #
        # ==================== #        
        class_loss = self.mse(
            torch.flatten(exists_box* predictions[..., :20], end_dim=-2),
            torch.flatten( exists_box * target[..., :20], end_dim = -2)
        )
        loss = (
            self.lambda_coord*box_loss + pred_loss + self.lambda_noobj*class_loss
        )
        return loss