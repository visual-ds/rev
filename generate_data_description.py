import os 
import glob 
import re 

types = ["academic", "quartz", "vega"] 
root = "data" 
labels = ["predicted", "bbs", "mask", "debug"]   

def main(): 
    for cls in types: 
        print("Generating data for", cls, end = "\r", flush = True) 
        description = open(root + "/" + cls + ".txt", "w") 
        data_dir = root + "/" + cls 

        # for academic and quartz, the images 
        # aren't compartimentalized in folders 
        if cls in types[:2]: 
            files = os.listdir(data_dir) 
            for img in files: 
                is_raw = ".png" in img 
                for label in labels: 
                    is_raw = is_raw and (label not in img) 

                if is_raw: 
                    description.write("./data/" + cls + "/" + img + "\n") 
        else: 
            folders = glob.glob(data_dir + "/*") 
            for folder in folders: 
                print("Current folder:", folder) 
                files = os.listdir(folder) 
                # print("files", files) 
                for img in files: 
                    # print(img) 
                    is_raw = ".png" in img 
                    for label in labels: 
                        is_raw = is_raw and (label not in img) 

                    if is_raw: 
                        description.write(folder[10:] + "/" + img + "\n") 

if __name__ == "__main__": 
    main() 
