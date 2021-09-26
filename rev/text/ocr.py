# -*- coding: utf-8 -*-
#from cv2 import cv2
import cv2
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

# packages for deep text recognition
import string
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

from models.dlocr.utils import AttnLabelConverter
from models.dlocr.dataset import RawDataset, AlignCollate, TextBoxDataset
from models.dlocr.model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    api = PyTessBaseAPI(psm=psm, lang='eng', path = "/usr/share/tesseract-ocr/4.00/tessdata/")
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

        border_size = 3
        roi_bw = cv2.copyMakeBorder(roi_bw,
                 border_size,
                 border_size,
                 border_size,
                 border_size,
                 cv2.BORDER_CONSTANT,
                 value=255
        )

        #u.show_image('ex', roi_bw)

        pil_img = Image.fromarray(roi_bw)

        max_conf = -np.inf

        correct_text = ''
        correct_angle = 0

        for angle in [0, -90, 90]:
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
            #print('hiiiiiiiiiiiii')
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

def deep_ocr(args, opt, text_boxes, chart_image):

    if None in args.values():
        raise KeyError(f"The parameters {list(args.keys())} should be available!")

    args = {**args, **opt}

    converter = AttnLabelConverter(args["character"])

    args["num_class"] = len(converter.character)

    if opt["rgb"]:
        opt["input_channel"] = 3

    model = Model(args)

    print("Model input parameters", args, opt)

    model = torch.nn.DataParallel(model).to(device)

    # load model
    print("loading pretrained model from", args["saved_model"])

    model.load_state_dict(torch.load(args["saved_model"], map_location = device))

    # prepare data
    align = AlignCollate(imgH = args["imgH"], imgW = args["imgW"], keep_ratio_with_pad = args["PAD"])
    # data = RawDataset(root = args["image_folder"], opt = args)
    data = TextBoxDataset(chart_image, text_boxes, args)
    loader = torch.utils.data.DataLoader(
        data, batch_size = args["batch_size"],
        shuffle = False,
        num_workers = int(args["workers"]),
        collate_fn = align, pin_memory = True
    )

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in loader:
            # print(image_tensors)
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            length_for_pred = torch.IntTensor([opt["batch_max_length"]] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, args["batch_max_length"] + 1).fill_(0).to(device)

        # u.show_image(image, "a")
        preds = model(image, text_for_pred, is_train = False)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

        log = str()
        preds_prob = F.softmax(preds, dim = 2)
        preds_max_prob, _ = preds_prob.max(dim = 2)

        # boxes = dict()

        for box in text_boxes:
            box._text_conf = -np.inf

        angles = data.angles

        for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
            pred_EOS = pred.find("[s]")
            pred = pred[:pred_EOS] # [s] is the end of sentence token
            pred_max_prob = pred_max_prob[:pred_EOS]
            # print(img_name)

            box_idx, angle = list(map(int, img_name.split("|")))

            box = text_boxes[box_idx]

            confidence_score = pred_max_prob.cumprod(dim = 0)[-1]

            # angle = angles[angle]

            if confidence_score > box._text_conf:
                box._text = pred
                box._text_conf = confidence_score
                box._text_dist = 1e-9
                box._text_angle = angle
                # print(box._text, box._text_conf)

            # print(box._text, box._text_conf)
            # print(pred, confidence_score)
            # dashed = "~" * 99
            # print(dashed)

            log = log + f"{img_name:25s}\t{pred:25s}\t{confidence_score:.4f}"

    return text_boxes



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
