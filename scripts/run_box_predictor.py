#!/usr/bin/env python
"""
Script to create label figures using actual bounding boxes

Usage:
  run_box_predictor.py single INPUT_PNG [--debug]
  run_box_predictor.py multiple INPUT_LIST_TXT [--debug]
  run_box_predictor.py (-h | --help)
  run_box_predictor.py --version

Options:
  -o OUTPUT   Output file.
  --debug     Show debug image.
  -h --help   Show this screen.
  --version   Show version.
"""
import os
import sys

sys.path.append(".")
os.environ['OMP_THREAD_LIMIT'] = '1'


from docopt import docopt
from joblib import Parallel, delayed
import multiprocessing


#from sklearn.utils import parallel_backend
from joblib import parallel_backend

#import chartprocessor.box_predictor as bp
from rev.text.localizer import TextLocalizer


#from chartprocessor.chart import chart_dataset, Chart
from rev.chart import Chart, chart_dataset


from tesserocr import PyTessBaseAPI, PSM
import cv2
from PIL import Image


def single(chart, debug = False): 
    localizer = TextLocalizer(method='default')
    chart.text_boxes = localizer.localize([chart], debug=debug)[0]
    chart.save_text_boxes()
    chart.save_debug_image()

if __name__ == '__main__':
    args = docopt(__doc__, version='1.0')
    debug = False
    if args['--debug']:
        debug = True

    if args['single']:
        image_name = args['INPUT_PNG']

        localizer = TextLocalizer(method='default')
        chart = Chart(image_name, text_from=2)
        single(chart, debug)


    if args['multiple']:
        chart_list = args['INPUT_LIST_TXT']

        num_cores = multiprocessing.cpu_count()

        with parallel_backend('loky'):
          results = Parallel(verbose=10, n_jobs=num_cores)(delayed(single)(chart) for chart in chart_dataset(chart_list, from_bbs=2))
