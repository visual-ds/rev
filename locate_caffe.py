import os 
import caffe 

if __name__ == "__main__": 
    caffe_path = caffe.__file__ 
    caffe_path = caffe_path[:caffe_path.rindex("/")] 
    print(caffe_path) 
