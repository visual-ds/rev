""" 
Usage: 
    python rate_craft.py CHART_TYPE CUDA  

CHART_TYPE must be one of `academic` or `quartz`; 
they correspond to the source of our data. 
Also, there must be a `{CHART_TYPE}.txt` file 
in the data directory; it contains the list 
of the pngs.  
"""

import os 
import sys 
import gc 
import torch 

sys.path.append(".") 

from rev.chart import Chart 
from rev.text.localizer import TextLocalizer 

craft_model = "./models/craft/craft_mlt_25k.pth" 
data = "data/" 
types = ["academic", "quartz", "vega", "debug"]  

cuda = torch.cuda.is_available() and (len(sys.argv) > 2 and sys.argv[2])   

def run_predictor(chart_list, root_dir): 
    for chart in chart_list: 
        print("Prediction for", chart.strip()) 
        localizer = TextLocalizer("craft", 
                craft_model = craft_model) 
        chart = Chart(chart.strip(), text_from = 2) 
        
        try: 
            chart.text_boxes = localizer.localize([chart], cuda = cuda)[0] 
            chart.save_text_boxes() 
            chart.save_debug_image() 
        except: 
            logfile.write(chart.filename) 

        gc.collect() 
        torch.cuda.empty_cache() 

if __name__ == "__main__": 
    args = sys.argv 
     
    chart_class = args[1] 
    if chart_class not in types: 
        raise ValueError(f"arg 1 should be in {types}") 
    
    if chart_class != "debug": 
        logfile = open("debug.txt", "w") 

    root_dir = data + chart_class 
    chart_list = open(root_dir + ".txt").readlines() 
    run_predictor(chart_list, root_dir + "/") 
    logfile.close() 
