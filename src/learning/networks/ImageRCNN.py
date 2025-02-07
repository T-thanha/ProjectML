from torch import nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights

class FasterRCNN(nn.Module):
    def __init__(self, num_classes=2):  
        super(FasterRCNN, self).__init__()
        self.model = fasterrcnn_resnet50_fpn_v2(pretrained=True,weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    def forward(self, x, targets=None):
        return self.model(x, targets)