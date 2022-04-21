import cv2
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator.build import build_proposal_generator
from detectron2.structures import Boxes, ImageList, Instances, RotatedBoxes
from detectron2.utils.events import EventStorage


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)

    cfg.MODEL.DEVICE = 'cpu'
    cfg.TEST.DETECTIONS_PER_IMAGE = 2000
    cfg.MODEL.WEIGHTS = "../models/model_tt_r101.pth"

    # Set model
    # cfg.MODEL.WEIGHTS = args.weights
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


# https://detectron2.readthedocs.io/en/latest/tutorials/models.html#partially-execute-a-model
# https://github.com/facebookresearch/detectron2/issues/5
# https://medium.com/@hirotoschwert/digging-into-detectron-2-part-2-dd6e8b0526e
def _seg():
    torch.manual_seed(121)
    cfin = get_cfg()
    cfin.confidence_threshold = .7
    cfin.config_file = "../configs/ocr/totaltext_101_FPN.yaml"
    cfg = setup_cfg(cfin)
    model = build_model(cfg)
    model.eval()
    # checkpointer = DetectionCheckpointer(model)
    # checkpointer.load(cfg.MODEL.WEIGHTS, from_last_checkpoint=False)
    backbone = model.backbone
    print(backbone)
    image = cv2.imread("../input_images/snippet-001.png")
    # 1, Get W and H of raw image input
    height, width = image.shape[:2]
    print("Original image size: ", (height, width))

    # image = self.transform_gen.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    inputs = [{"image": image, "height": height, "width": width}]
    images = model.preprocess_image(inputs)  # get a ImageList of Detectron2

    # 3, Get feature of backbone via FPN
    features = model.backbone(images.tensor)
    print("Backbone Features:", features)  # a dict for backbone feature (p2, p3, p4, p5, p6) layers
    # ['p2', 'p3', 'p4', 'p5', 'p6']
    # [1, 256, 48, 176]
    p2 = features['p2']
    print(features.keys())
    # NCHW
    print(features['p2'].shape)  # stride = 4
    print(features['p3'].shape)  # stride = 8
    print(features['p4'].shape)  # stride = 16
    print(features['p5'].shape)  # stride = 32
    print(features['p6'].shape)  # stride = 64

    for i in range(256):
        p2 = features['p2'][0][i].cpu().detach().numpy() * 255
        cv2.imwrite(f"../features/features_p2_{i}.png", p2)

if __name__ == "__main__":
    _seg()
