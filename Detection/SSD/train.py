import torch

def box_nms(bboxes, scores, threshold=0.5):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break
            
        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(min=x2[i].item())
        yy2 = y2[order[1:]].clamp(min=y2[i].item())

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w*h
        overlap = inter / (areas[i] + areas[order[1:]] - inter)

        ids = (overlap <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.tensor(keep, dtype=torch.long)


def train(train_loop_func, logger, args):
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    cocoGt = get_coco_ground_truth(args)

    train_dataset = get_train_dataset(args)
    train_loader = get_train_dataloader(train_dataset, args)

    val_dataset = get_val_dataset(args)
    val_loader = get_val_dataloader(val_dataset, args)

    ssd300 = SSD300(backbone=ResNet(args.backbone, args.backbone_path))
    args.learning_rate = args.learning_rate * args.N_gpu * (args.batch_size / 32)
    start_epoch = 0
    iteration = 0
    loss_func = Loss(dboxes)

    if use_cuda:
        ssd300.cuda()
        loss_func.cuda()
    
    optimizer = torch.optim.SGD(tencent_trick(ssd300), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)

    inv_map = {v:k for k, v in val_dataset.lable_map.items()}
    
    total_time = 0

    if args.mode == 'evaluation':
        acc = evaluate(ssd300, val_dataloader, cocoGT, encoder, inv_map, args)
        print('Model precision {} mAP'.format(acc))
        return