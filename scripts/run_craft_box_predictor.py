#!usr/bin/env python
"""
Script for labelling text regions of a given image using CRAFT's CNN
implementation.

Usage:
    run_craft_box_predictor.py single INPUT_PNG [--model=MODEL_NAME] [--debug]
    run_craft_box_predictor.py multiple INPUT_LIST_TXT [--model=MODEL_NAME] [--debug]
    run_craft_box_predictor.py (-h | --help)
    run_craft_box_predictor.py --version

Options:
    --model=MODEL_NAME  Model's location
    --debug  Show debug image
    -h --help  Show this screen
    --version  Show current version
"""
import sys

sys.path.append(".")

from rev.chart import Chart
from rev.text.localizer import TextLocalizer

from docopt import docopt

def single(chart, localizer):
    debug = args["--debug"]
    chart.text_boxes = localizer.localize([chart], debug = debug)[0]

    print("Text boxes:",
    chart.text_boxes, sep = "\n")

def multiple(charts, localizer):
    debug = args["--debug"]
    text_boxes = localizer.localize(charts, debug = debug)
    print("Text boxes:",
    text_boxes, sep = "\n")
if __name__ == "__main__":

    args = docopt(__doc__, version = "1.0")

    model_path = args["--model"]

    if args["single"]:
        # model_path = args["--model"]
        localizer = TextLocalizer(method = "craft",
            craft_model = model_path)

        image_path = args["INPUT_PNG"]
        chart = Chart(image_path)
        single(chart, localizer)

    if args["multiple"]:
        # model_path = args["--model"]
        localizer = TextLocalizer(method = "craft",
            craft_model = model_path)

        input_list = args["INPUT_LIST_TXT"]

        charts = []

        with open(input_list, "r") as file:
            lines = file.readlines()
            charts = [Chart(line.rstrip()) for line in lines]

        multiple(charts, localizer)
