#!/usr/bin/env python
"""
Rate the predicted bounding boxes.

Usage:
    rate_box_predictor.py INPUT_LIST_TXT (--mask | --perfect | --overlap) [--from_bbs=FROM] [--pad=PAD] [--debug]
    rate_box_predictor.py (-h | --help)
    rate_box_predictor.py --version

Options:
    --mask            Use masks for rating.
    --perfect         Use Maximum Bipartite Matching for rating.
    --overlap         Use overlap rule for rating.
    --from_bbs FROM   1: from predicted1-bbs.csv
                      2: from predicted2-bbs.csv  [default: 1]
    --pad PAD         Add padding to boxes [default: 0]
    --debug           Show debug image.
    -h --help         Show this screen.
    --version         Show version.
"""
import sys
sys.path.append(".")

from docopt import docopt
import numpy as np
import cv2

from rev.third_party import metric_binary as mb
from rev.chart import chart_dataset, create_mask
from rev import utils as u
from rev.text import rectutils as ru
from rev.third_party.mwmatching import maxWeightMatching

from joblib import Parallel, delayed
import multiprocessing

DEBUG = False


def rate_boxes_using_masks(chart, from_bbs, pad=0):
    fh, fw, _ = chart.image.shape
    truth_boxes = chart.text_boxes

    # u.show_image("chart", chart.image)

    pred_boxes = chart.get_text_boxes(text_from = from_bbs)

    if pad > 0:
        for b in pred_boxes:
            b._rect = ru.wrap_rect(b._rect, fh, fw, padx=pad, pady=pad)


    if pad > 0:
        for b in pred_boxes:
            b.rect = ru.wrap_rect(b._rect, fh, fw, padx=pad, pady=pad)

    truth = create_mask(fh, fw, truth_boxes)
    pred = create_mask(fh, fw, pred_boxes)

    # threshold to get bw image
    _, pred = cv2.threshold(pred, 128, 255, cv2.THRESH_BINARY)
    _, truth = cv2.threshold(truth, 128, 255, cv2.THRESH_BINARY)

    dice = mb.dc(pred, truth)
    jaccard = mb.jc(pred, truth)
    precision = mb.precision(pred, truth)
    recall = mb.recall(pred, truth)
    f1score = (2.0 * precision * recall / (precision + recall))

    if DEBUG:
        cv2.imshow('truth', truth)
        cv2.imshow('pred', pred)

        cv2.moveWindow('truth', 100, 10)
        cv2.moveWindow('pred', 300, 10)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return dice, jaccard, precision, recall, f1score


def rate_boxes_using_perfect_matching(chart, from_bbs, pad):
    target_boxes = chart.text_boxes
    estima_boxes = chart.get_text_boxes(text_from = from_bbs)
    fh, fw, _ = chart.image.shape

    # increase a the box size because our boxes are very tight
    if pad > 0:
        for b in estima_boxes:
            b._rect = ru.wrap_rect(b._rect, fh, fw, padx=pad, pady=pad)

    if DEBUG:
        truth_dbg = chart.image

        debug_chart = chart.copy(text_from = from_bbs)
        predicted_dbg = debug_chart.debug()

        graph_img = np.concatenate((truth_dbg, predicted_dbg), axis=1)
        matching_img = np.concatenate((truth_dbg, predicted_dbg), axis=1)

    # creating bipartite graph
    edges = []
    n = len(target_boxes)
    for i, box1 in enumerate(target_boxes):
        for j, (box2, coeff) in enumerate(box1.find_best_match(estima_boxes, return_all=True)):
            edges.append((i, j+n, coeff))

            if DEBUG:
                p1 = box1.center()
                p2 = box2.center()
                cv2.line(graph_img, u.ttoi(p1), u.ttoi((p2[0] + fw, p2[1])), (0, 0, 256))
                cv2.line(graph_img, u.ttoi(p1), u.ttoi((p2[0] + fw, p2[1])), (0, 0, 256))

    if DEBUG:
        cv2.imshow('Graph', graph_img)
        cv2.moveWindow('Graph', 0, 0)

    # compute perfect matching
    mate = maxWeightMatching(edges)

    coeffs = []
    for i in range(n):
        if mate[i] == -1:
            continue

        box1 = target_boxes[i]
        box2 = estima_boxes[mate[i] - n]

        coeffs.append(box1.matching_score(box2))

        if DEBUG:
            p1 = box1.center()
            p2 = box2.center()
            cv2.line(matching_img, u.ttoi(p1), u.ttoi((p2[0] + fw, p2[1])), (0, 0, 256))

    if DEBUG:
        cv2.imshow('Matching', matching_img)
        cv2.moveWindow('Matching', 0, fh + 50)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return np.mean(coeffs), 0


def rate_boxes_using_overlap(chart, from_bbs, pad=0):

    target_boxes = chart.text_boxes
    estima_boxes = chart.get_text_boxes(text_from = from_bbs)

    # increase a the box size because our boxes are very tight
    # if pad > 0:
    fh, fw, _ = chart.image.shape
    for b in estima_boxes:
        b._rect = ru.wrap_rect(b._rect, fh, fw, padx=pad, pady=pad)

    precision = 0.0
    for ebox in estima_boxes:
        match = ebox.find_best_match(target_boxes)
        # if match[1] > 0.5:
        #     precision += 1.0
        precision += match[1]

    precision /= len(estima_boxes)

    recall = 0.0
    for tbox in target_boxes:
        match = tbox.find_best_match(estima_boxes)
        # if match[1] > 0.5:
        #     recall += 1.0
        recall += match[1]
    recall /= len(target_boxes)

    f1score = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)

    error = 0
    if f1score != 1.0:
        error = 1
        # print chart.predicted_debug_name(from_bbs)
        # print 'P=%0.4f R=%0.4f F=%0.4f' % (precision, recall, f1score)

    if DEBUG:
        truth_dbg = chart.image()
        predicted_dbg = chart.predicted_debug(from_bbs)

        graph_img = np.concatenate((truth_dbg, predicted_dbg), axis=1)
        cv2.imshow('Graph', graph_img)
        cv2.moveWindow('Graph', 0, 0)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return precision, recall, f1score, error



def main(args):
    chart_list = args['INPUT_LIST_TXT']
    from_bbs = int(args['--from_bbs'])
    pad = int(args['--pad'])


    num_cores = multiprocessing.cpu_count()
    if args['--debug']:
        num_cores = 1

    # data = chart_dataset(chart_list)

    if args["--debug"]:
        for chart in data:
            print(chart.image)


    if args['--mask']:
        # run in parallel
        results = Parallel(n_jobs=num_cores, verbose=1, backend='multiprocessing')(
            delayed(rate_boxes_using_masks)(chart, from_bbs, pad) for chart in chart_dataset(chart_list, 2))

        coeffs = np.asarray(results)
        print('Dice     : %0.2f' % coeffs[:, 0].mean())
        print('Jaccard  : %0.2f' % coeffs[:, 1].mean())
        print('Precision: %0.2f' % coeffs[:, 2].mean())
        print('Recall   : %0.2f' % coeffs[:, 3].mean())
        print('F1-Score : %0.2f' % coeffs[:, 4].mean())
        
        dataset = chartlist[chartlist.index(".")] 
        filename = datetime.today().strftime("%Y-%m-%d") 
        filename = "metrics-" + filename + ".csv" 
        
        metrics = coeffs.mean(axis = 0) 

        with open(filename, "w") as file: 
            data = f""" 
dataset,metric,method,value
{dataset},dice,_,{metrics[0]}  
{dataset},jaccard,_,{metrics[1]}
{dataset},precision,_,{metrics[2]}
{dataset},recall,_,{metrics[3]}
{dataset},f1,_,{metrics[4]}
""" 
            file.write(data) 

    if args['--perfect']:
        # run in parallel
        results = Parallel(n_jobs=num_cores, verbose=1, backend='multiprocessing')(
            delayed(rate_boxes_using_perfect_matching)(chart, from_bbs, pad) for chart in chart_dataset(chart_list))

        coeffs = np.asarray(results)
        print('Jaccard     : %0.2f' % coeffs[:, 0].mean())


    if args['--overlap']:
        # run in parallel
        results = Parallel(n_jobs=num_cores, verbose=1, backend='multiprocessing')(
            delayed(rate_boxes_using_overlap)(chart, from_bbs, pad) for chart in chart_dataset(chart_list))

        coeffs = np.asarray(results)
        print('Precision: %0.4f' % coeffs[:, 0].mean())
        print('Recall   : %0.4f' % coeffs[:, 1].mean())
        print('F1-Score : %0.4f' % coeffs[:, 2].mean())
        wrong = coeffs[:, 3].sum()
        total = len(results)
        print('#Wrong Charts: %d/%d (%0.2f%%)' % (wrong, total, wrong / total * 100.0))


if __name__ == '__main__':

    args = docopt(__doc__, version='1.0')


    if args['--debug']:
        DEBUG = True


    main(args)
