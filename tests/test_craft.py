import os 
import torch 

cuda = torch.cuda.is_available() 

from rev.chart import Chart 
from rev.text.localizer import TextLocalizer 
from rev.text.classifier import TextClassifier 
from rev.spec.generator import SpecGenerator 

import json 

# Parameters 
attn_parameters = {
        "saved_model": "../models/attn/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth" 
} 

craft_params = {
        "text_threshold": .7, 
        "link_threshold": .4, 
        "low_text": .4, 
        "poly": False, 
        "canvas_size": 1280, 
        "mag_ratio": 1.8, 
        "cuda": cuda 
} 

def test_spec_generation(path_chart_image: str, 
        ground_truth: dict): 
    """ 
    Check whether the implementation is consistent; it applies, for this, 
    the pipeline in the bitmap image in `path_chart_image` and confront 
    the generated specification with the `ground_truth`. 
    """ 
    assert os.path.exists(path_chart_image), \
            "The file {path_chart_image} doesn't exist!".format(path_chart_image = path_chart_image) 

    # Instantiate chart 
    chart = Chart(path_chart_image) 

    # Use CRAFT to detect text boxes 
    localizer = TextLocalizer("craft", 
            craft_model = "../models/craft/craft_mlt_25k.pth", 
            ocr = "attn", 
            attn_params = attn_parameters) 
    
    chart.text_boxes = localizer.localize([chart])[0] 

    # Classify text boxes 
    text_classifier = "../models/text_role_classifier/text_type_classifier.pkl" 
    text_clf = TextClassifier(model_checkpoint = text_classifier) 
    
    text_type_preds = text_clf.classify([chart]) 

    for (text_box, role) in zip(chart.text_boxes, text_type_preds[0]): 
        text_box.type = role 

    # Generate specification 
    spec_gen = SpecGenerator() 
    spec = spec_gen.generate([chart]) 
    
    assert spec[0] == ground_truth
    return spec 

if __name__ == "__main__": 
    chart_path = "../examples/image.png" 
    ground_truth = json.load(open("craft.json")) 
    test_spec_generation(chart_path, ground_truth) 




