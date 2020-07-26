from collections import OrderedDict
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

from . import util


class Scale:
    def __init__(self, channel, label_boxes):
        self.channel = channel
        self.label_boxes = label_boxes

        # quantitative: linear|pow|exp
        # temporal: time|ordinal
        # ordinal: ordinal
        # nominal: nominal
        self.type = ''
        self.domain = []
        self.range = []

        if not self.label_boxes:
            return

        # infer scale
        self.infer_ranges()

        # infer scale type
        self.infer_scale_type()

    def infer_ranges(self):
        # if quantitative: [value_1, value_2] -> [pos_1, pos_2]
        if self.channel.type == 'quantitative':

            first_box, last_box = self.label_boxes[0], self.label_boxes[-1]
            x1, y1 = first_box['xc'], first_box['yc']
            x2, y2 = last_box['xc'], last_box['yc']
            t1, t2 = first_box['number'], last_box['number']

            if self.channel.orientation == 'x':
                self.domain = [
                    t1, t2
                ]
                self.range = [
                    int(util.continuous_mapping(t1, t1, t2, x1, x2)),
                    int(util.continuous_mapping(t2, t1, t2, x1, x2))
                ]
            else:
                self.domain = [t1, t2]
                self.range = [
                    int(util.continuous_mapping(t1, t1, t2, y1, y2)),
                    int(util.continuous_mapping(t2, t1, t2, y1, y2))
                ]

        # if ordinal : [] -> []

        # if temporal:  [] -> []
        if self.channel.type == 'temporal':
            self.domain = [box['text'] for box in self.label_boxes]
            self.range = []
            for box in self.label_boxes:
                self.range.append(int(box['xc']) if self.channel.orientation == 'x' else box['yc'])

        # if nominal: [label1, .., labeln] -> [pos_1, ..., pos_n]
        if self.channel.type == 'nominal':
            self.domain = [box['text'] for box in self.label_boxes]
            self.range = []
            for box in self.label_boxes:
                self.range.append(int(box['xc']) if self.channel.orientation=='x' else box['yc'])

    def infer_scale_type(self):
        if self.channel.type == 'quantitative':
            scales = [
                {
                    'name': 'linear',
                    # 'func': lambda x: x,
                    'ifunc': lambda x, m, b: m * x + b
                },
                {
                    'name': 'log',
                    # 'func': lambda x: np.ma.log10(x),
                    'ifunc': lambda x, m, b: m * np.power(10, x) + b
                },
                {
                    'name': 'pow',
                    # 'func': lambda x: np.power(x, 2),
                    'ifunc': lambda x, m, b: m * np.power(x, 0.5) + b
                },
                {
                    'name': 'sqrt',
                    # 'func': lambda x: np.sqrt(x),
                    'ifunc': lambda x, m, b: m * np.power(x, 2) + b
                }
            ]

            if self.channel.orientation == 'x':
                xdata = np.array([box['xc'] for box in self.label_boxes])
            else:
                length = self.channel.spec_heigth
                xdata = np.array([length - box['yc'] for box in self.label_boxes])

            ydata = np.array([box['number'] for box in self.label_boxes])

            # hack to avoid errors with log scale
            mapping = np.vectorize(util.continuous_mapping, otypes=[np.float])
            xdata = mapping(xdata, np.min(xdata), np.max(xdata), 1, 10)

            pred_scale = 'linear'
            min_error = np.inf
            for scale in scales:
                ifunc = scale['ifunc']

                # reset data
                x = xdata
                y = ydata

                # removing zeros if log scale
                if scale['name'] == 'log':
                    valid_idx = x > 0
                    x = x[valid_idx]
                    y = y[valid_idx]

                try:
                    # fit a line
                    popt, pcov = curve_fit(ifunc, x, y)
                    error = mean_squared_error(y, ifunc(x, *popt))
                except Exception:
                    error = np.inf

                # saving best option
                if error < min_error:
                    pred_scale = scale['name']
                    min_error = error

            self.type = pred_scale

        if self.channel.type == 'nominal':
            self.type = 'nominal'

    def gen(self):
        scl = OrderedDict()
        scl['type'] = self.type
        scl['labels'] = [b['text'] for b in self.label_boxes]
        if self.channel.type == 'quantitative':
            scl['values'] = [b['number'] for b in self.label_boxes]
        if self.channel.type == 'temporal':
            scl['values'] = [b['number'] for b in self.label_boxes]
        scl['domain'] = self.domain
        scl['range'] = self.range
        return scl