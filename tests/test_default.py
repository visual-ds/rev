import os 
import sys 

import json 

from rev.spec.generator import SpecGenerator 
from rev.chart import Chart 
from rev.text.localizer import TextLocalizer 
from rev.text import TextClassifier 

def test_spec_generation(path_chart_image: str, 
        ground_truth: dict): 
    """ 
    Make a confrontation between the spec generation in `path_chart_image` and the 
    `ground_truth`, which is the dictionary available in `default.json`. 
    """ 
    assert os.path.exists(path_chart_image), \
        "The directory {path_chart_image} doesn't exist!".format(
                path_chart_image = path_chart_image
                )   
    classifier_model = "../models/text_role_classifier/text_type_classifier.pkl" 

    # Instantiate chart 
    chart = Chart(path_chart_image) 
    
    # Localize texts 
    localizer = TextLocalizer(ocr = "tesseract") 
    text_boxes = localizer.localize([chart]) 
    chart.text_boxes = text_boxes[0] 

    # Classify texts 
    text_clf = TextClassifier(model_checkpoint = classifier_model) 
    text_type_preds = text_clf.classify([chart]) 

    # Set the role attribute for each text box 
    for (text_box, role) in zip(chart.text_boxes, text_type_preds[0]): 
        text_box.type = role 

    # Generate specification 
    spec_gen = SpecGenerator() 
    spec = spec_gen.generate([chart]) 

    assert spec[0] == ground_truth 

    return spec 

if __name__ == "__main__": 
    ground_truth = json.load(open("default.json", "r")) 
    chart_path = "../examples/image.png" 
    test_spec_generation(chart_path, 
            ground_truth) 
    

    




