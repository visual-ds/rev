import numpy as np
from PIL import ImageDraw, ImageFont, Image
import os
import cv2

CWD = os.path.dirname(os.path.abspath(__file__))

unicode_font = ImageFont.truetype(os.path.join(CWD, '../fonts/Arial Unicode.ttf'), 12)


text_colors = {
    'x-axis-title': '#3182bd',
    'x-axis-label': '#9ecae1',
    'y-axis-title': '#e6550d',
    'y-axis-label': '#fdae6b',
    'legend-title': '#31a354',
    'legend-label': '#a1d99b',
    'text-label':   '#FFFF00',
    'title':        '#FF0000',
    '-':            '#aaaaaa',
    '': '#000000'
}



def rgba2rgb(img):
    """
    Convert the rgba image into a rgb with white background.
    :param img:
    :return:
    """
    arr = img.astype('float') / 255.
    alpha = arr[..., -1]
    channels = arr[..., :-1]
    out = np.empty_like(channels)

    background = (1, 1, 1)
    for ichan in range(channels.shape[-1]):
        out[..., ichan] = np.clip(
            (1 - alpha) * background[ichan] + alpha * channels[..., ichan],
            a_min=0, a_max=1)

    return (out * 255.0).astype('uint8')


def ttoi(t):
    """
    Converts tuples values to tuple of rounded integers.
    """
    return tuple(map(int, map(round, t)))


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def create_predicted1_bbs(chart):
    """
    Create an empty bbs file with empty texts and types.
    :param chart:
    :return:
    """
    import rev
    ifn = chart.filename.replace('.png', '-texts.csv')
    text_boxes = rev.chart.load_texts(ifn)

    # cleaning type field
    for text_box in text_boxes:
        text_box._type = ''

    ofn = chart.filename.replace('.png', '-pred1-texts.csv')
    rev.chart.save_texts(text_boxes, ofn)


def draw_rects(img, rects, color=(0, 0, 255), thickness=1, invert=True):
    if invert:
        img = 255-img
    if not isinstance(rects[0], tuple):
        rects = [box._rect for box in rects]

    for x, y, w, h in rects:
        cv2.rectangle(img, ttoi((x, y)), ttoi((x+w-1, y+h-1)), color, thickness=thickness)

    return img


def draw_graph(img, graph, vertices, thickness=1, color=(0, 255, 255)):
    for vi, vj in graph.edges():
        ci, cj = vertices[vi], vertices[vj]
        cv2.line(img, ttoi(ci), ttoi(cj), color=color, thickness=thickness)

    return img



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def draw_boxes(img, boxes):
    #vis = smp.toimage(img).convert('RGBA')
    vis = Image.fromarray(img).convert('RGBA')

    rects = Image.new('RGBA', vis   .size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(rects)
    for b in boxes:
        outline = tuple(reversed(hex_to_rgb(text_colors[b.type])))
        fill = outline + (80,)
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=outline, fill=fill)

    for b in boxes:
        draw.text((b.x, b.y - 16), b.text, fill=(255, 0, 0), font=unicode_font)

    #return smp.fromimage(Image.alpha_composite(vis, rects))
    return np.asarray(Image.alpha_composite(vis, rects))