import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import random
import itertools
import json
import time
import bz2
import pickle
import cv2
from math import sqrt
from augmentations import SSDAugmentation, ToCV2Image

COCO_ROOT = 'data/coco'
IMAGES = 'images'
ANNOTATIONS = 'annotations'
INSTANCES_SET = 'instances_{}.json'
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')

def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map

class COCOAnnotationTransform(object):
    '''
    Transforms a COCO annotation into a Tensor of bbox corrds and label index
    '''

    def __init__(self):
        self.label_map = get_label_map(os.path.join(COCO_ROOT, 'coco_labels.txt'))
    
    def __call__(self, target, width, height):
        '''
        Inputs:
            target(dict): COCO target json annotation as a python dict
            height(int)
            width(int)
        Outputs:
            a list containing lists of bounding boxes [bbox coords, class idx]
        '''
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array(bbox) / scale)
                final_box.append(label_idx)
                res += [final_box]
            else:
                print('no bbox problem!')
        return res

class COCODetection(data.Dataset):
    def __init__(self, root, image_set='train2014', transform=None,
                target_transform=COCOAnnotationTransform(), dataset_name='MS COCO'):
        from pycocotools.coco import COCO
        self.root = os.path.join(root, IMAGES, image_set)
        self.coco = COCO(os.path.join(root, ANNOTATIONS, INSTANCES_SET.format(image_set)))
        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        '''
        Inputs:
            index(int): Index
        Outputs:
            tuple: Tuple(image, target).
                target is the object returned by coco.loadAnns
        '''
        img_id = self.ids[index]
        img_path = os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        assert os.path.exists(img_path), 'Image path does not exist: {}'.format(img_path)
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        ann_ids = self.coco.getAnnIds(imgIds = img_id)
        target = self.coco.loadAnns(ann_ids)
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
    
    def get_image(self, index):
        img_id = self.ids[index]
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(os.path.join(self.root, img_path), cv2.IMREAD_COLOR)
    
    def get_anno(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds = img_id)
        return self.coco.loadAnns(ann_ids)
    
    def __repr__(self):
        fmt_str = 'Dataset' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints:{}\n'.format(self.__len__())
        fmt_str += '    Root Location:{}\n'.format(self.root)
        fmt_str += '    Transforms:{}\n'.format(self.transform.__repr__())
        fmt_str += '    Target Transforms:{}\n'.format(self.target_transform.__repr__())
        return fmt_str


def calc_iou_tensor(box1, box2):
    '''
    input:
        box1 (N, 4)
        box2 (M, 4)
    output:
        IoU(N, M)
    '''
    N = box1.size(0)
    M = box2.size(0)

    be1 = box1.unsqueeze(1).expand(-1, M, -1)
    be2 = box2.unsqueeze(0).expand(N, -1, -1)
    lt = torch.max(be1[:, :, :2], be2[:, :, :2])
    rb = torch.min(be1[:, :, 2:], be2[:, :, 2:])
    delta = rb - lt
    delta[delta < 0] = 0
    intersect = delta[:, :, 0] * delta[:, :, 1]
    delta1 = be1[:, :, 2:] - be1[:, :, :2]
    area1 = delta1[:, :, 0] * delta1[:, :, 1]
    delta2 = be2[:, :, 2:] - be2[:, :, :2]
    area2 = delta2[:, :, 0] * delta2[:, :, 1]

    iou = intersect / (area1 + area2 - intersect)
    return iou

class Encoder(object):
    '''
    Transform between (bboxes, labels) <-> SSD output
    dboxes: defalut boxes in size 8732 x 4
        encoder: input ltrb format, output xywh format
        decoder: input xywh format, output ltrb format

    encode:
        input   : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
        output  : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
        criteria : IoU threshold of bboxes

    decode:
        input   : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
        output  : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
        criteria : IoU threshold of bboxes
        max_output : maximum number of output bboxes
    '''

    def __init__(self, dboxes):
        self.dboxes = dboxes(order='ltrb')
        self.dboxes_xywh = dboxes(order='xywh').unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0)
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh
    
if __name__ == '__main__':
    dataset = COCODetection(root=COCO_ROOT, transform=SSDAugmentation())
    #dataset = COCODetection(root=COCO_ROOT, transform=None)
    print(dataset)
    for i in range(len(dataset)):
        image, targets, height, width = dataset[i]
        image = image.permute(1, 2, 0).numpy()
        image = image[:, :, (2, 1, 0)]
        image = np.ascontiguousarray(image, dtype=np.uint8)
        print("image {}, {}x{}".format(i, height, width))

        for target in targets:
            l = target[0] * width
            t = target[1] * height
            r = target[2] * width
            b = target[3] * height
            label = target[4]
            print("BBox ({}, {}) - ({}, {})".format(int(l), int(t), int(r), int(b)))
            image = cv2.rectangle(image, (int(l), int(t)), (int(r), int(b)), (0, 255, 255), 2)

        ori_image = dataset.get_image(i)
        ori_annos = dataset.get_anno(i)
        for anno in ori_annos:
            if 'bbox' in anno:
                bbox = anno['bbox']
                x, y, w, h = bbox
                ori_image = cv2.rectangle(ori_image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 255), 2)
        
        #hori = np.concatenate((image, ori_image), axis=1)
        cv2.imshow('Demo', image)
        cv2.waitKey(5000)
