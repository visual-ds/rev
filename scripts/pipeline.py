from rev.chart import Chart
from rev.text.localizer import TextLocalizer
from rev.text.classifier import TextClassifier
from rev.spec.generator import SpecGenerator

import json
import torch 

attn_parameters = { 
    # "image_folder": "imgboxes", 
    "saved_model": "../models/attn/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth", 
} 

cuda = torch.cuda.is_available() 

craft_params = {
    "text_threshold": .7,
    "link_threshold": .4,
    "low_text": .4,
    "poly": False,
    "canvas_size": 1280,
    "mag_ratio": 1.8,
    "cuda": cuda 
} 

# import torch
chart = Chart("../data/quartz/B1LOWeGM.png")

# cuda = torch.cuda.is_available()

text_classifier = {
    "default": "../models/text_role_classifier/text_type_classifier.pkl"
}

localizer = TextLocalizer("craft",
                          craft_model = "../models/craft/craft_mlt_25k.pth",
                          craft_params = craft_params,
                         ocr = "attn",
                         attn_params = attn_parameters)

chart.text_boxes = localizer.localize([chart],
                                      debug = True)[0]

# print(chart.text_boxes)

text_clf = TextClassifier(model_checkpoint = text_classifier["default"])
text_type_preds = text_clf.classify([chart])

# Set the role for each textbox on the chart
for (text_box, role) in zip(chart.text_boxes, text_type_preds[0]):
    text_box.type = role

# Generate specification and chart mark
spec_gen = SpecGenerator()
spec = spec_gen.generate([chart])

json.loads(spec[0])
