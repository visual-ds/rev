#import cv2
#from cv2 import cv2
import cv2
import os
import math
import itertools

from .. chart import Chart
from .. textbox import TextBox
from .. import utils as u
from . import rectutils as ru
from . import ocr

from . pixel_link_text_detector import text_detect, PixelLinkDetector

from scipy.ndimage.morphology import binary_propagation
from scipy.spatial.distance import pdist, squareform, euclidean

from skimage import morphology
from skimage.measure import regionprops
from skimage.color import label2rgb

import networkx as nx
import numpy as np

from numpy import random



#temporal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



class TextLocalizer:
    def __init__(self, method = 'default'):
        self._method = method

        if self._method == 'pixel_link':
            self._pixel_link_detector = PixelLinkDetector()
            self._pixel_link_detector.init()


    def default_localize(self, charts, preproc_scale = 1.5, debug=False):

        lsboxes = []

        for chart in charts:
            img = chart.image

            # pre-processing
            img = cv2.resize(img, None, fx=preproc_scale, fy=preproc_scale, interpolation=cv2.INTER_CUBIC)
            fh, fw, _ = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)


            # remove non-text regions
            bw_text = apply_mask(bw, chart.mask)

            # complete text-regions components
            bw_rec = binary_propagation(bw_text, mask=bw)
            bw_rec = bw_rec.astype('uint8') * 255
            connected_comp, num_comp = morphology.label(bw_rec, return_num=True)
            regions = regionprops(connected_comp, cache=True)
            boxes = filter_regions(regions, preproc_scale)

            if debug:
                image_label_overlay = label2rgb(connected_comp, image=bw_rec, bg_label=0, bg_color=(1, 1, 1), alpha=1)
                show_image('connected components', image_label_overlay, 400, 300)
                vis = u.draw_rects(cv2.cvtColor(bw_rec, cv2.COLOR_GRAY2BGR), boxes, thickness=2, color=(0, 0, 255))
                show_image('characters', vis, 800, 300)

            # merging characters
            boxes = merge_characters(img, bw_rec, boxes, preproc_scale, debug)

            # Apply OCR and filter by confidence and filter
            boxes = ocr.run_ocr_in_boxes(img, boxes, pad=3, psm=8) #8 for single word
            min_conf = 25
            max_dist = 4
            boxes = [box for box in boxes if box._text_conf > min_conf and box._text_dist < max_dist]
            min_conf = 40
            boxes = [box for box in boxes if box._text_conf > min_conf]

            if debug:
                vis = u.draw_rects(cv2.cvtColor(bw_rec, cv2.COLOR_GRAY2BGR), boxes, color=(0, 0, 255), thickness=2)
                show_image('after ocr conf', vis, 400, 600)

            # merge words
            boxes = merge_words(img, boxes)
            if debug:
                vis = u.draw_rects(cv2.cvtColor(bw_rec, cv2.COLOR_GRAY2BGR), boxes, color=(0, 0, 255), thickness=2)
                show_image('after merging words', vis, 800, 600)

            vis = u.draw_boxes(img.copy(), boxes)

            # recover original image
            for b in boxes:
                b._rect = [d / preproc_scale for d in b._rect]
            vis2 = u.draw_boxes(chart.image, boxes)

            if debug:
                show_image('text', vis, 1200, 600)
                show_image('original', vis2, 0, 900)

            lsboxes.append(boxes)
        
        return lsboxes

    def pixel_link_localize(self, charts, debug=False):

        img_paths = [chart.filename for chart in charts]
        lspoints = self._pixel_link_detector.predict_multiple(img_paths)
        lsboxes = []

        for index, chart in enumerate(charts):

            points = lspoints[index]

            if debug:
                img_temp = chart.image.copy()
                for i, bbox in enumerate(points):
                    pts = np.array([[bbox[0],bbox[1]],[bbox[2],bbox[3]],[bbox[4],bbox[5]],[bbox[6],bbox[7]]], np.int32)
                    pts = pts.reshape((-1,1,2))
                    cv2.polylines(img_temp,[pts],True,(0,0,255))

                    cv2.putText(img_temp, str(i), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
                
                show_image('pixel_link bboxes', img_temp)

        
            points = [ e.reshape(-1,2) for e in np.array(points)]

        
            # Apply OCR and filter by confidence and filter
            img = chart.image.copy()

            boxes = ocr.run_ocr_in_points_boxes(img, points, pad=3, psm=8, debug=False)
            boxes2 = []
            for i, point in enumerate(points):
                xmin = min(point, key = lambda t: t[0])[0]
                ymin = min(point, key = lambda t: t[1])[1]
                xmax = max(point, key = lambda t: t[0])[0]
                ymax = max(point, key = lambda t: t[1])[1]
                boxes2.append(TextBox(i, xmin, ymin, xmax-xmin, ymax-ymin))

            #boxes = ocr.run_ocr_in_boxes(img, boxes2, pad=3, psm=8, debug=True) #8 for single word
            

            if debug:
                img_temp = chart.image.copy()
                img_temp = u.draw_boxes(img_temp, boxes)
                show_image('bboxes from points after ocr', img_temp)


            if debug:
                img_temp = chart.image.copy()
                img_temp = u.draw_boxes(img_temp, boxes2)
                show_image('bboxes from points', img_temp)


            #boxes = ocr.run_ocr_in_boxes(img, boxes, pad=3, psm=8) #8 for single word

            #min_conf = 25
            #boxes = [box for box in boxes if box._text_conf > min_conf]

            #min_conf = 25
            #max_dist = 4
            #boxes = [box for box in boxes if box._text_conf > min_conf and box._text_dist < max_dist]
            #min_conf = 40
            #boxes = [box for box in boxes if box._text_conf > min_conf]

            boxes = merge_words(img, boxes)

            vis = u.draw_boxes(img.copy(), boxes)

            if debug:
                show_image('merged bboxes', vis)

            lsboxes.append(boxes)
    
        return lsboxes
        

    def localize(self, charts, debug=False):
        
        if self._method == 'default':
            return self.default_localize(charts, 1.5, debug)

        elif self._method == 'pixel_link':
            return self.pixel_link_localize(charts, debug)
        else:
            raise Exception('wrong "method" parameter, only supports: "default" or "pixel_link"')



# functions
def apply_mask(bw, pred):

    #print(pred)

    h, w = bw.shape
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    _, mask = cv2.threshold(pred, 60, 255, cv2.THRESH_BINARY)

    return cv2.bitwise_and(bw, bw, mask=mask)


def filter_regions(regions, scale):
    boxes = []
    for i, region in enumerate(regions):
        minr, minc, maxr, maxc = region.bbox
        x, y, w, h = u.ttoi((minc, minr, maxc - minc, maxr - minr))

        # filter by aspect ratio
        if not (1 / 15 < w / float(h) < 15):
            continue

        # filter by size
        if not (4 * scale * scale <= w * h < 1000 * scale * scale):
            continue

        # if region.eccentricity > 0.995:
        #     continue
        #
        # if region.solidity < 0.3:
        #     continue
        #
        # if region.euler_number < -4:
        #     continue
        #
        # if 0.2 > region.extent or region.extent > 0.9:
        #     continue

        box = TextBox(i, x, y, w, h)
        box._regions.append(region)
        boxes.append(box)

    return boxes


def show_image(name, image, x=0, y=0):
    imgplot = plt.imshow(image)
    plt.gcf().canvas.set_window_title(name)
    plt.show()
    
    #new_name = '%s - %s - [%0.2f, %0.2f] - %s' % (name, image.shape, image.min(), image.max(), image.dtype.name)
    #image = image.astype(np.uint8)
    #cv2.imshow(new_name, image)
    #cv2.moveWindow(new_name, x, y
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()




def should_be_removed(r1, r2, L, scale):
    c1, c2 = ru.center(r1), ru.center(r2)
    d12 = euclidean(c1, c2)

    xi, yi, _ = ru.rect_segment_intersection(r1, (c1, c2))
    d1 = euclidean(c1, (xi, yi))

    xj, yj, _ = ru.rect_segment_intersection(r2, (c1, c2))
    d2 = euclidean(c2, (xj, yj))

    # distance between rectangles that is not inside the rect
    delta = d12 - d1 - d2

    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    vert_overlap, vert_len = ru.range_overlap(x1, x1 + w1, x2, x2 + w2)
    hori_overlap, hori_len = ru.range_overlap(y1, y1 + h1, y2, y2 + h2)
    is_vert_align = vert_len > min(w1, w2) * 0.5
    is_horz_align = hori_len > min(h1, h2) * 0.5

    w1 = w1 if w1 > 6 * scale else L
    w2 = w2 if w2 > 6 * scale else L
    h1 = h1 if h1 > 6 * scale else L
    h2 = h2 if h2 > 6 * scale else L

    # if is_horz_align and delta < (w1 + w2) / 2.0:
    #     return False
    # if is_vert_align and delta < (h1 + h2) / 2.0:
    #     return False
    if is_horz_align and delta < min(w1, w2) * 0.5:
        return False
    if is_vert_align and delta < min(h1, h2) * 0.5:
        return False

    return True



def merge_characters(img, bw, boxes, scale,  debug = False):
    fh, fw, _ = img.shape

    # compute minimum spanning tree
    centers = [box.center() for box in boxes]
    graph = nx.from_numpy_matrix(squareform(pdist(centers)))
    mst = nx.minimum_spanning_tree(graph)

    if debug:
        vis = u.draw_rects(cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR), boxes, thickness=2, color=(0, 0, 255))
        vis = u.draw_graph(vis, mst, centers, thickness=2, color=(0, 204, 0))
        show_image('mst', vis, 1200, 300)

    lengths = list(itertools.chain(*[(box.w, box.h) for box in boxes]))
    # lengths = np.array(rects)[:, 2:4].ravel()
    counts = np.bincount(lengths)
    common_height = np.argmax(counts)

    weights = [mst[vi][vj]['weight'] for (vi, vj) in mst.edges()]
    counts = np.bincount(weights)
    common_length = np.argmax(counts)

    # initial pass to ignore points
    #for n, nbrs in mst.adjacency_iter():
    for n, nbrs in mst.adjacency():
        b = boxes[n]
        min_area = 16
        if mst.degree(n) == 2 and b.area() < min_area:
            n1, n2 = tuple(nbrs.keys())
            mst.remove_edge(n, n1)
            mst.remove_edge(n, n2)
            mst.add_edge(n1, n2)

    for v1, v2 in mst.edges():
        r1, r2 = boxes[v1]._rect, boxes[v2]._rect

        # filter by location
        if should_be_removed(r1, r2, common_length, scale):
            mst.remove_edge(v1, v2)
            continue

        # Filter edges by color
        th_color = 20
        if ru.color_dist(img, bw, r1, r2) > th_color:
            mst.remove_edge(v1, v2)
            continue

    # prune edges related with nodes with degree 4
    #for n, nbrs in mst.adjacency_iter():
    for n, nbrs in mst.adjacency():
        if mst.degree(n) == 4:
            longest_dist = -1
            longest_vertex = None
            for nbr, eattr in nbrs.items():
                c1, c2 = centers[n], centers[nbr]
                dist = euclidean(c1, c2)
                if dist > longest_dist:
                    longest_dist = dist
                    longest_vertex = nbr

            mst.remove_edge(n, longest_vertex)

    # prune edges related with nodes with degree 3
    #for n, nbrs in mst.adjacency_iter():
    for n, nbrs in mst.adjacency():
        if mst.degree(n) == 3:
            horz, vert = [], []
            for nbr, eattr in nbrs.items():
                c1, c2 = centers[n], centers[nbr]
                angle = math.degrees(math.atan2(c2[1] - c1[1], c2[0] - c1[0]))
                if abs(angle) <= 30 or 0 <= 180 - abs(angle) <= 30:
                    horz.append(nbr)
                elif abs((abs(angle) - 90)) <= 30:
                    vert.append(nbr)
                else:
                    assert True

            to_remove = horz if len(horz) < len(vert) else vert
            for nbr in to_remove:
                mst.remove_edge(n, nbr)

    # prune edges related with nodes with degree 2
    #for n, nbrs in mst.adjacency_iter():
    for n, nbrs in mst.adjacency():
        if mst.degree(n) == 2:
            n1, n2 = tuple(nbrs.keys())
            c, c1, c2 = centers[n], centers[n1], centers[n2]
            angle = math.degrees(u.angle_between((c1[0]-c[0], c1[1]-c[1]), (c2[0]-c[0], c2[1]-c[1])))

            b, b1, b2 = boxes[n], boxes[n1], boxes[n2]
            min_area = 16 * scale * scale
            #if abs(c[0]-212) < 1.0 and (c[1]-42) < 1.0:
                #print(c)
            a, a1, a2 = b.area(), b1.area(), b2.area()
            if angle < 120 and a > min_area and a1 > min_area and a2 > min_area:
                # keep the closest
                if euclidean(c, c1) > euclidean(c, c2):
                    mst.remove_edge(n, n1)
                else:
                    mst.remove_edge(n, n2)

    new_boxes = []
    for i, indices in enumerate(nx.connected_components(mst)):
        box = TextBox.merge_boxes([boxes[idx] for idx in indices], i)
        new_boxes.append(box)
    boxes = new_boxes

    # Final check if a rectangle is inside another
    graph = nx.empty_graph(len(boxes))
    for i, b1 in enumerate(boxes):
        for j, b2 in enumerate(boxes):
            if i == j or ru.inside(b1._rect, b2._rect):
                graph.add_edge(i, j)

    new_boxes = []
    for i, indices in enumerate(nx.connected_components(graph)):
        box = TextBox.merge_boxes([boxes[idx] for idx in indices], i)
        new_boxes.append(box)
    boxes = new_boxes

    if debug:
        vis = u.draw_graph(cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR), mst, centers, thickness=1)
        vis = u.draw_rects(vis, boxes)
        show_image('mst_filtered', vis, 0, 600)

    return boxes


def merge_words(img, boxes):
    graph = nx.empty_graph(len(boxes))
    for i, b1 in enumerate(boxes):
        h1 = b1.h if b1._text_angle == 0 else b1.w
        for j, b2 in enumerate(boxes):
            h2 = b2.h if b2._text_angle == 0 else b2.w

            is_horizontal = b1._text_angle == 0 and b2._text_angle == 0
            same_angle = abs(b1._text_angle) == abs(b2._text_angle)
            same_height = ru.same_height(b1._rect, b2._rect, horiz=is_horizontal)
            near = ru.next_on_same_line(b1._rect, b2._rect, dist=min(h1, h2), horiz=is_horizontal)

            if i == j or (same_angle and same_height and near):
                graph.add_edge(i, j)

    new_boxes = []
    for i, indices in enumerate(nx.connected_components(graph)):
        if len(indices) > 1:
            box = TextBox.merge_boxes([boxes[idx] for idx in indices], i)
            ocr.run_ocr_in_boxes(img, [box], pad=3)
            new_boxes.append(box)
        else:
            box = boxes[indices.pop()]
            box.id = i
            new_boxes.append(box)

    # Final check if a rectangle is inside another
    graph = nx.empty_graph(len(new_boxes))
    for v1 in graph.nodes():
        for v2 in graph.nodes():
            if ru.inside(new_boxes[v1]._rect, new_boxes[v2]._rect):
                graph.add_edge(v1, v2)

    new_boxes2 = []
    for i, indices in enumerate(nx.connected_components(graph)):
        if len(indices) > 1:
            box = TextBox.merge_boxes([boxes[idx] for idx in indices], i)
            new_boxes2.append(box)
        else:
            box = new_boxes[indices.pop()]
            box.id = i
            new_boxes2.append(box)

    return new_boxes2