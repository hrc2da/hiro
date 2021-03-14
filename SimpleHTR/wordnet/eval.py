import argparse
from collections import namedtuple

import numpy as np
import torch
from path import Path

from wordnet.aabb import AABB
from wordnet.aabb_clustering import cluster_aabbs
from wordnet.coding import decode, fg_by_cc
from wordnet.dataloader import DataLoaderIAM
from wordnet.dataset import DatasetIAM, DatasetIAMSplit, DatasetHIRO, DatasetHIROSplit
from wordnet.iou import compute_dist_mat_2
from wordnet.loss import compute_loss
from wordnet.net import WordDetectorNet
from wordnet.utils import compute_scale_down
from wordnet.visualization import visualize_and_plot

EvaluateRes = namedtuple('EvaluateRes', 'batch_imgs,batch_aabbs,loss,metrics')


class BinaryClassificationMetrics:
    def __init__(self, tp, fp, fn):
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def accumulate(self, other):
        tp = self.tp + other.tp
        fp = self.fp + other.fp
        fn = self.fn + other.fn
        return BinaryClassificationMetrics(tp, fp, fn)

    def recall(self):
        return self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else 0

    def precision(self):
        return self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else 0

    def f1(self):
        re = self.recall()
        pr = self.precision()
        return 2 * pr * re / (pr + re) if pr + re > 0 else 0


def binary_classification_metrics(gt_aabbs, pred_aabbs):
    iou_thres = 0.7

    ious = 1 - compute_dist_mat_2(gt_aabbs, pred_aabbs)
    match_counter = (ious > iou_thres).astype(np.int)
    gt_counter = np.sum(match_counter, axis=1)
    pred_counter = np.sum(match_counter, axis=0)

    tp = np.count_nonzero(pred_counter == 1)
    fp = np.count_nonzero(pred_counter == 0)
    fn = np.count_nonzero(gt_counter == 0)

    return BinaryClassificationMetrics(tp, fp, fn)


def infer_one(net, img, thres=0.5, max_aabbs=None):
    with torch.no_grad():
        y = net(img, apply_softmax=True)
        pred_map = y.to('cpu').numpy()[0]
        scale_up = 1 / compute_scale_down(net.input_size, net.output_size)
        aabbs = decode(pred_map, comp_fg=fg_by_cc(thres, max_aabbs), f=scale_up)
        h, w = img.to('cpu').numpy()[0][0].shape
        aabbs = [aabb.clip(AABB(0, w - 1, 0, h - 1)) for aabb in aabbs]
        clustered_aabbs = cluster_aabbs(aabbs)
    return clustered_aabbs 

def evaluate(net, loader, thres=0.5, max_aabbs=None):
    batch_imgs = []
    batch_aabbs = []
    loss = 0

    for i in range(len(loader)):
        # get batch
        loader_item = loader[i]
        with torch.no_grad():
            y = net(loader_item.batch_imgs, apply_softmax=True)
            y_np = y.to('cpu').numpy()
            if loader_item.batch_gt_maps is not None:
                loss += compute_loss(y, loader_item.batch_gt_maps).to('cpu').numpy()

        scale_up = 1 / compute_scale_down(WordDetectorNet.input_size, WordDetectorNet.output_size)
        metrics = BinaryClassificationMetrics(0, 0, 0)
        # import pdb; pdb.set_trace()
        for i in range(len(y_np)):
            img_np = loader_item.batch_imgs[i, 0].to('cpu').numpy()
            pred_map = y_np[i]

            aabbs = decode(pred_map, comp_fg=fg_by_cc(thres, max_aabbs), f=scale_up)
            h, w = img_np.shape
            aabbs = [aabb.clip(AABB(0, w - 1, 0, h - 1)) for aabb in aabbs]  # bounding box must be inside img
            clustered_aabbs = cluster_aabbs(aabbs)

            if loader_item.batch_aabbs is not None:
                curr_metrics = binary_classification_metrics(loader_item.batch_aabbs[i], clustered_aabbs)
                metrics = metrics.accumulate(curr_metrics)

            batch_imgs.append(img_np)
            batch_aabbs.append(clustered_aabbs)

    return EvaluateRes(batch_imgs, batch_aabbs, loss / len(loader), metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--data_dir', type=Path, required=True)
    args = parser.parse_args()

    net = WordDetectorNet()
    net.load_state_dict(torch.load('../model/weights'))
    net.eval()
    net.to('cuda')

    dataset = DatasetHIRO(args.data_dir, net.input_size, net.output_size, caching=False)
    dataset_eval = DatasetHIROSplit(dataset, 0, 10)
    loader = DataLoaderIAM(dataset_eval, args.batch_size, net.input_size, net.output_size)

    res = evaluate(net, loader, max_aabbs=1000)
    print(f'Loss: {res.loss}')
    print(f'Recall: {res.metrics.recall()}')
    print(f'Precision: {res.metrics.precision()}')
    print(f'F1 score: {res.metrics.f1()}')

    for img, aabbs in zip(res.batch_imgs, res.batch_aabbs):
        visualize_and_plot(img, aabbs)


if __name__ == '__main__':
    main()
