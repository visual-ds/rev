from ..chart import Chart
from ..textbox import TextBox
from ..mark.classifier import MarkClassifier
from . import channel

from collections import OrderedDict

from typing import List

import json


class SpecGenerator:
    def __init__(self):
        self._mark_classifier = MarkClassifier()

    
    def filter(self, text_type, text_boxes:List[TextBox]):
        boxes = [box for box in text_boxes if box._type == text_type]

        # sorting axis labels
        if text_type == 'x-axis-label':
            boxes = sorted(boxes, key=lambda b: b.xc)

        elif text_type == 'y-axis-label':
            boxes = sorted(boxes, key=lambda b: b.yc, reverse=True)

        return boxes
    
    def generate(self, charts:List[Chart]):

        mark_types = self._mark_classifier.classify(charts)

        ls_specs = [] 

        for index, chart in enumerate(charts):
            spec = OrderedDict()

            h, w, _ = chart.image.shape
            spec['width'] = w
            spec['height'] = h
            title_boxes = self.filter('title', chart.text_boxes)
            spec['title'] = '|'.join( [t.text for t in title_boxes] )
            spec['mark'] = mark_types[index]
            spec['encoding'] = self.encoding(chart)
            spec = json.dumps(spec)
            ls_specs.append(spec)

        return ls_specs


    def encoding(self, chart):
        enc = OrderedDict()
        
        h, w, _ = chart.image.shape

        x_title_boxes = self.filter('x-axis-title', chart.text_boxes)
        x_label_boxes = self.filter('x-axis-label', chart.text_boxes)
        x_chanel = channel.Channel(w, h, x_title_boxes, x_label_boxes, 'x')

        y_title_boxes = self.filter('y-axis-title', chart.text_boxes)
        y_label_boxes = self.filter('y-axis-label', chart.text_boxes)
        y_chanel = channel.Channel(w, h, y_title_boxes, y_label_boxes, 'y')

        enc['x'] = x_chanel.gen()
        enc['y'] = y_chanel.gen()

        return enc

        

            
             

        