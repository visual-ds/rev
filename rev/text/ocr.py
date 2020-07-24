# -*- coding: utf-8 -*-
from cv2 import cv2
import os
import re
import numpy as np

import scipy.misc as smp

from tesserocr import PyTessBaseAPI, PSM

from skimage import morphology
from skimage.segmentation import clear_border

from .. import utils as u
from . import rectutils as ru

from ..textbox import TextBox

# temporal

# from chartprocessor.chart import save_bbs
# from chartprocessor.third_party.textconvert import lossy_unicode_to_ascii
# import chartprocessor.utils as u

from ..third_party.textconvert import lossy_unicode_to_ascii

from PIL import Image


SHOW = False


def post_process_text(text):
    # '1O99' -> '1099'
    # '4o' -> '40'
    # 'l' -> '1'
    # print('joaooooooooooo ', text, '   ', type(text))
    text = text.decode("utf-8")
    tmp = text.replace('O', '0')
    tmp = tmp.replace('o', '0')
    tmp = tmp.replace('l', '1')
    if u.is_number(tmp):
        text = tmp

    # '1 5%' -> '15%'
    # '51 ,050' -> '51,050'
    # '1001 -5000' -> '1001-5000'
    pos = text.find('1 ')
    while pos != -1:
        skip = 2

        # 'Q1 14' -> 'Q1 14'
        if pos - 1 >= 0 and text[pos - 1] in '0OQ':
            # pos = text.find('1 ', pos + 2)
            skip = 2
            # continue

        # '1 999' -> '1999'
        if text[pos + 2].isdigit() or text[pos + 2] in ',.-':
            text = text[:pos + 1] + text[pos + 2:]
            skip = 1

        pos = text.find('1 ', pos + skip)

    # In Quartz is common to use Q1, Q2, Q3 and Q4
    # ex. '02' -> 'Q2'
    # ex. '02 14' -> 'Q2 14'
    if re.match(r'0\d(?:\s|$)', text):
        text = 'Q' + text[1:]

    # '°/o of keynote' -> '% of keynote'
    text = text.replace('°/o', '%')

    return text


def run_ocr_in_points_boxes(img, point_set, pad=0, psm=PSM.SINGLE_LINE, debug=False):

    # add a padding to the initial figure
    fpad = 1
    img = cv2.copyMakeBorder(img.copy(), fpad, fpad, fpad,
                             fpad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    fh, fw, _ = img.shape

    api = PyTessBaseAPI(psm=psm, lang='eng')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    boxes = []


    for i, points in enumerate(point_set):


        roi = u.four_point_transform(img, points, 1)


        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.resize(roi_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, roi_bw = cv2.threshold(
        roi_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        #if i in [14]:
        #u.show_image('ex', roi_bw)

        pil_img = Image.fromarray(roi_bw)
        
        max_conf = -np.inf
        
        correct_text = ''
        correct_angle = 0
        
        for angle in [0, 90, 180, 270]:
            rot_img = pil_img.rotate(angle, expand=1)
            api.SetImage(rot_img)
            conf = api.MeanTextConf()
            text = api.GetUTF8Text().strip()

            if debug:
                u.show_image('{id}_{angle}_{conf}_{text}'.format(id=i, angle=angle, conf=conf, text=text), rot_img)

            if conf > max_conf:
                max_conf = conf
                correct_text = text
                correct_angle = angle

        text = post_process_text(lossy_unicode_to_ascii(correct_text))
        text_conf = max_conf
        text_angle = correct_angle

        xmin = min(points, key = lambda t: t[0])[0]
        ymin = min(points, key = lambda t: t[1])[1]
        xmax = max(points, key = lambda t: t[0])[0]
        ymax = max(points, key = lambda t: t[1])[1]

        boxes.append(TextBox(i, xmin, ymin, xmax-xmin, ymax-ymin, text=text, text_conf=text_conf, text_angle=text_angle ))
        # u.log('num comp %d' % box.num_comp)
        # u.log(u'** text: {} conf: {} angle: {}'.format(correct_text, max_conf, correct_angle))

    api.End()

    return boxes



def run_ocr_in_boxes(img, boxes, pad=0, psm=PSM.SINGLE_LINE, debug = False):
    """
    Run OCR for all the boxes.
    :param img:
    :param boxes:
    :param pad: padding before applying ocr
    :param psm: PSM.SINGLE_WORD or PSM.SINGLE_LINE
    :return:
    """
    # add a padding to the initial figure
    fpad = 1
    img = cv2.copyMakeBorder(img.copy(), fpad, fpad, fpad, fpad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    fh, fw, _ = img.shape

    api = PyTessBaseAPI(psm=psm, lang='eng')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    for box in boxes:
        # adding a pad to original image. Some case in quartz corpus, the text touch the border.
        x, y, w, h = ru.wrap_rect(u.ttoi(box._rect), fh, fw, padx=pad, pady=pad)
        x, y = x + fpad, y + fpad

        if w * h == 0:
            box.text = ''
            continue

        # crop region of interest
        roi = img[y:y + h, x:x + w]
        #  to gray scale
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #
        roi_gray = cv2.resize(roi_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        # binarization
        _, roi_bw = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # removing noise from borders
        roi_bw = 255 - clear_border(255-roi_bw)

        # roi_gray = cv2.copyMakeBorder(roi_gray, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)

        # when testing boxes from csv files
        if box.num_comp == 0:
            # Apply Contrast Limited Adaptive Histogram Equalization
            roi_gray2 = clahe.apply(roi_gray)
            _, roi_bw2 = cv2.threshold(roi_gray2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            _, num_comp = morphology.label(roi_bw2, return_num=True, background=255)
            box._regions.extend(range(num_comp))

        # pil_img = smp.toimage(roi_bw)
        pil_img = Image.fromarray(roi_bw)

        if SHOW:
            pil_img.show()
        max_conf = -np.inf
        min_dist = np.inf
        correct_text = ''
        correct_angle = 0
        for angle in [0, -90, 90]:
            rot_img = pil_img.rotate(angle, expand=1)

            

            api.SetImage(rot_img)
            conf = api.MeanTextConf()
            text = api.GetUTF8Text().strip()
            dist = abs(len(text.replace(' ', '')) - box.num_comp)

            if debug:
                u.show_image('{id}_{angle}_{conf}_{text}'.format(id = box._id, angle = angle, conf = conf, text = text), rot_img)

            # u.log('text: %s  conf: %f  dist: %d' % (text, conf, dist))
            if conf > max_conf and dist <= min_dist:
                max_conf = conf
                correct_text = text
                correct_angle = angle
                min_dist = dist

        box._text = post_process_text(lossy_unicode_to_ascii(correct_text))
        box._text_conf = max_conf
        box._text_dist = min_dist
        box._text_angle = correct_angle

        # u.log('num comp %d' % box.num_comp)
        # u.log(u'** text: {} conf: {} angle: {}'.format(correct_text, max_conf, correct_angle))

    api.End()

    return boxes


# def run_ocr_in_chart(chart, from_bbs, pad=0):
#     """
#     Run OCR for all the boxes in a chart and save them in a csv file.
#     :param chart:
#     :param from_bbs: 1: from predicted1-bbs.csv
#                      2: from predicted2-bbs.csv  [default: 1]
#     :return:
#     """
#     assert (from_bbs != 0)
#     if from_bbs == 1 and not os.path.isfile(chart.predicted_bbs_name(1)):
#             u.create_predicted1_bbs(chart)

#     boxes = chart.predicted_bbs(from_bbs)
#     run_ocr_in_boxes(chart.image(), boxes, pad=pad)

#     bb_name = chart.predicted_bbs_name(from_bbs)
#     save_bbs(bb_name, boxes)
