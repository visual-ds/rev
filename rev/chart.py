#from cv2 import cv2
import cv2
import pandas as pd
import os.path
import numpy as np

from .textbox import TextBox
from . import utils as u
from .text.mask_predictor import predict_mask

import copy
'''
    The attribute 'text_from' means:
    0: read ground truth data:
        'chart-texts.csv'
        'chart-mask.png'
        'chart-debug.png'  
    1: read text from 'prediction 1':
        i.e. ground truth boxes and output of text role classification and output of OCR.
        'chart-pred1-texts.csv'
        'chart-pred1-mask.png'
        'chart-pred1-debug.png'
    2: read text from 'prediction 2', 
        i.e., output of text localization and output of text role classification, and output of OCR.
        'chart-pred2-texts.csv'
        'chart-pred2-mask.png'
        'chart-pred2-debug.png'
'''
prefixes = {0: '', 1: '-pred1', 2: '-pred2'}


class Chart(object):
    def __init__(self, fn, _id=None, text_from=0):
        self._fn = fn
        self._id = _id
        self._text_from = text_from
        self._image = None
        self._texts = None
        self._mark_type = None

        self._prefix = prefixes[text_from]

    @property
    def filename(self):
        return self._fn

    @property
    def text_boxes_filename(self):
        return self._fn.replace('.png', self._prefix + '-texts.csv')

    @property
    def id(self):
        return self._id

    @property
    def text_from(self):
        return self._text_from

    @property
    def image(self):
        if self._image is None:
            print
            self._image = cv2.imread(self._fn, cv2.IMREAD_UNCHANGED)
            if self._image.dtype == 'uint16':
                self._image = (self._image / 256.0).astype('uint8')
            else:
                self._image = self._image.astype('uint8')

            if len(self._image.shape) == 2:
                self._image = cv2.merge((self._image, self._image, self._image))
            elif self._image.shape[2] == 4:
                self._image = u.rgba2rgb(self._image)
                cv2.imwrite(self._fn, self._image)

        return self._image

    @property
    def text_boxes(self):
        if self._texts is None:
            fn = self._fn.replace('.png', self._prefix + '-texts.csv')
            self._texts = load_texts(fn)

        return self._texts

    @text_boxes.setter
    def text_boxes(self, values):
        self._texts = values

    # setting the textboxes values     
    def set_text_boxes(self, texts):
        self._texts = texts 

    # setting the textboxes values     
    def update_prefix(self, text_from):
        self._prefix = prefixes[text_from] 

    @property
    def mask(self, force_to_create=False):
        fn = self._fn.replace('.png', self._prefix + '-mask.jpg')
        if not os.path.exists(fn) or force_to_create:
            predict_mask(self)
        
        return cv2.imread(fn, cv2.IMREAD_GRAYSCALE)


    @property
    def debug(self):
        fn = self._fn.replace('.png', self._prefix + '-debug.png')
        return cv2.imread(fn, cv2.IMREAD_COLOR)


    def save_text_boxes(self):
        save_texts(self.text_boxes, self.text_boxes_filename)

    def get_text_boxes(self, text_from = 0):
        fn = self._fn.replace('.png', prefixes[text_from] + '-texts.csv')
        return load_texts(fn)

    def save_debug_image(self):
        #if not os.path.exists(self.predicted_debug_name):
        debug_image = u.draw_boxes(self.image, self.text_boxes)
        cv2.imwrite(self.predicted_debug_name, debug_image)
        
    @property
    def predicted_debug_name(self):
        return self._fn.replace('.png', '{prefix}-debug.png'.format(prefix=self._prefix))

    def copy(self, text_from = 0):
        new_chart = copy.deepcopy(self)
        new_chart.update_prefix(text_from)
        return new_chart



def create_mask(h, w, texts):
    mask = np.zeros((h, w), np.uint8)
    for t in texts:
        cv2.rectangle(mask, u.ttoi(t.p1), u.ttoi(t.p2), 255, thickness=-1)

    return mask


def load_texts(fn):
    try: 
        df = pd.read_csv(fn)
    except FileNotFoundError: 
        print("Using pred1", flush = True, end = "\r") 
        fn = fn.replace("texts", "pred1-texts") 
        df = pd.read_csv(fn) 

    df.replace(np.nan, '', inplace=True)

    # force text column to be string
    df.text = df.text.astype(str)

    texts = []
    for idx, row in df.iterrows():
        text = TextBox(row.id, row.x, row.y, row.width, row.height, row.type, row.text)
        texts.append(text)

    return texts


def save_texts(text_boxes, fn):
    rows = []
    for t in text_boxes:
    #print(text_boxes)
        rows.append(t.to_dict())
    df = pd.DataFrame(rows)
    df = df[rows[0].keys()]
    df.to_csv(fn, index=False)


def chart_dataset(chart_list, from_bbs=0):
    corpus = os.path.splitext(os.path.basename(chart_list))[0]
    with open(chart_list) as f:
        for idx, line in enumerate(f):
            yield Chart(line.strip(), _id='%s-%04d' % (corpus, idx), text_from=from_bbs)
