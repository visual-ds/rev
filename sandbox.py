import os
os.environ['DISPLAY'] = ':1'


import rev
from rev.text.localizer import TextLocalizer
# load a chart
chart = rev.Chart('examples/vega1.png', text_from=0)
myLocalizer = TextLocalizer(chart)
myLocalizer.localize(debug=True)