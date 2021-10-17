# Makefile for REV: it will, on the one 
# hand, install the dependencies (darknet 
# and Python packages); on the other hand, it 
# will configure the environment, writing 
# config.json and taking the models from 
# OSF 

ENV = env.yml 
CONFIG = `pwd`/./darknet

all: setenv darknet config models data
setenv:  
	conda env create -f $(ENV) 
darknet: 
	git clone https://github.com/visual-ds/darknet  
	cd darknet  
	make   
	cd ../. 
config: 
	echo {\
	\"darknet_lib_path\": \
	\"${CONFIG}\"\} \
	>> config.json   	
models: 
	!wget https://osf.io/ckj5z/download -O models.zip
	!unzip models.zip 
data: 
	!wget https://osf.io/uwtsv/download -O data.zip
	!unzip data.zip
