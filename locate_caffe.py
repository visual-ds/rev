import os 
import caffe 

""" 
Currently, we are installing 

caffe=1.0=py37hbab4207_5 

for caffe; however, it is inconsistency with other packages. In 
this context, we should modify some of its source code; specifically, 
the line 296 in the file caffe/io.py and the line 95 in caffe/classifier.py. The 
former, by the way, contains a line with `as_grey` (British English), so 
we need to replace it with `as_gray` (American English); the latter, on the 
other hand, applies float division (/) in a line in 
which in need integer division (//), so we treat this case! For details, please 
check the Makefile. 
""" 
if __name__ == "__main__": 
    caffe_path = caffe.__file__ 
    caffe_path = caffe_path[:caffe_path.rindex("/")] 
    print(caffe_path) 
