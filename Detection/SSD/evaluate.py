import torch
import time
import numpy as np
import io
from pycocotools.cocoeval import COCOeval

def evaluate(model, coco, cocoGt, encoder, inv_map, args):
    model.eval()
    ret = []
    start = time.time()

    #model = model.cuda()
    for nbatch, (img, img_id, img_size, _, _) in enumerate(coco):
        print("Parsing batch: {}/{}".format(nbatch, len(coco)), end='\r')
        with torch.no_grad():
            #inp = img.cuda()
            ploc, plabel = model(img)
            ploc, plabel = ploc.float(), plabel.float()

            for idx in range(ploc.shape[0]):
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)

                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, 0.50, 200)[0]
                except:
                    print("")
                    print("No object detected in idx: {}".format(idx))
                    continue
                    
                htot, wtot = img_size[0][idx].item(), img_size[1][idx].item()
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    ret.append([img_id[idx], loc_[0] * wtot,
                                loc_[1] * htot,
                                (loc_[2] - loc_[0]) * wtot,
                                (loc_[3] - loc_[1]) * htot,
                                prob_,
                                inv_map[label_]])

    ret = np.array(ret).astype(np.float32)
    final_results = ret
    print("")
    print("Predicting Ended, total time: {:.2f} s".format(time.time() - start))
    cocoDt = cocoGt.loadRes(final_results, use_ext=True)
    E = COCOeval(cocoGt, cocoDt, iouType='bbox', use_ext=True)
    E.evaluate()
    E.accumulate()
    E.summarize()
    print("Current AP: {:.5f}".format(E.stats[0]))

    model.train()

    return E.stats[0]