# Makefile for REV: it will, on the one 
# hand, install the dependencies (darknet 
# and Python packages); on the other hand, it 
# will configure the environment, writing 
# config.json and taking the models from 
# OSF 

ENV = env.yml 
CONFIG = `pwd`/darknet/./darknet
CONDA_ENV_NAME=rev

# Check conda. from https://stackoverflow.com/questions/60115420/check-for-existing-conda-environment-in-makefile 
 
ifeq (, $(shell which conda))
	HAS_CONDA=False 
else 
	HAS_CONDA=True
	ENV_DIR=$(shell conda info --base)
	MY_ENV_DIR=$(ENV_DIR)/envs/$(CONDA_ENV_NAME)
endif 


all: setenv caffe darknet config models data install
setenv:  
ifeq (True, $(HAS_CONDA))  
ifneq ("$(wildcard $(MY_ENV_DIR))", "") # Check whether the directory exists 
	@echo ">>> Found $(CONDA_ENV_NAME) environment in $(MY_ENV_DIR). Skipping installation." 
else 
	conda env create -f env.yml -n $(CONDA_ENV_NAME) 
endif 
else 
	@echo ">>> Install Conda!" 
	exit 
endif 
install: 
	pip install -e . 
darknet: 
	git clone https://github.com/visual-ds/darknet  
		cd darknet && make  
config: 
	echo {\
	\"darknet_lib_path\": \
	\"${CONFIG}\"\} \
	> config.json   	
models: 
	wget https://osf.io/ckj5z/download -O models.zip
	unzip models.zip 
data: 
	wget https://osf.io/uwtsv/download -O data.zip
	unzip data.zip
caffe: 
	sed -i '296s/.*/    img = skimage.img_as_float(skimage.io.imread(filename, as_gray=not color)).astype(np.float32)/' $(shell python locate_caffe.py)/io.py 
	sed -i '95s/.*/            predictions = predictions.reshape((len(predictions) \/\/ 10, 10, -1))/' $(shell python locate_caffe.py)/classifier.py
