#import os
#os.environ['DISPLAY'] = ':1'


import rev
from rev.text.localizer import TextLocalizer

# load a chart
chart = rev.Chart('examples/pmc1.png', text_from=0)
localizer = TextLocalizer()
text_boxes = localizer.localize(chart, method='pixel_link', debug=True)

new_chart = chart.copy(text_from=2)
new_chart.text_boxes = text_boxes
new_chart.save_text_boxes()
new_chart.save_debug_image()