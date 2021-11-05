#!/usr/bin/env python 

from distutils.core import setup 

setup(name = "rev", 
        version = "1.0", 
        description = "Reverse engineering visualizations", 
        author_email = "tdsh97@gmail.com", 
        url = "https://github.com/visual-ds/rev", 
        packages = ["rev"], 
        package_dir = {"rev": "rev"} 
) 
