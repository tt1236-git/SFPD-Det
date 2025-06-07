# Copyright (c) OpenMMLab. All rights reserved.
from .builder import ROTATED_DATASETS
from .dota import DOTADataset


@ROTATED_DATASETS.register_module()
class DIORRDataset(DOTADataset):
    """DIOR-R dataset for detection."""
    # DIOR-R数据集的20个类别
    CLASSES = (
        'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 
        'chimney', 'expressway-service-area', 'expressway-toll-station', 
        'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 
        'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 
        'vehicle', 'windmill'
    )

    PALETTE = [
        (0, 255, 0), (255, 0, 0), (138, 43, 226), (255, 128, 0), (255, 0, 255),
        (0, 255, 255), (165, 42, 42), (189, 183, 107), (255, 193, 193), 
        (0, 51, 153), (255, 250, 205), (0, 139, 139), (255, 255, 0), 
        (147, 116, 116), (0, 0, 255), (255, 85, 0), (255, 170, 0), 
        (128, 255, 0), (0, 128, 255), (255, 216, 0)
    ]