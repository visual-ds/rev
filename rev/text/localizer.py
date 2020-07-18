from .. chart import Chart, save_texts
from .. textbox import TextBox

from numpy import random

class TextLocalizer:
    def __init__(self, chart:Chart):

        self._chart = chart

    def localize(self):

        h, w, _ = self._chart.image.shape
        n_boxes = random.randint(1,15)
        textboxes = [ randombox(i, w, h) for i in range(n_boxes)]
        self._chart.set_text_boxes(textboxes)
        self._chart.update_prefix(2)
        save_texts(textboxes, self._chart.text_boxes_filename)


def randombox(id, w , h):
    return TextBox(id, random.randint(h), random.randint(w), w/random.randint(3,10), h/random.randint(3,10) )