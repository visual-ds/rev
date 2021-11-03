import numpy as np
from PIL import ImageDraw, ImageFont, Image
import os
#from cv2 import cv2
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


CWD = os.path.dirname(os.path.abspath(__file__))

unicode_font = ImageFont.truetype(
    os.path.join(CWD, 'Arial Unicode.ttf'), 12)


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
        cv2.rectangle(img, ttoi((x, y)), ttoi(
            (x+w-1, y+h-1)), color, thickness=thickness)

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

    # return smp.fromimage(Image.alpha_composite(vis, rects))
    return np.asarray(Image.alpha_composite(vis, rects))


def is_number(s):
    try:
        float(s)  # for int, long and float
    except ValueError:
        return False

    return True


# added by jaox

def show_image(name, image, x=0, y=0):
    imgplot = plt.imshow(image)
    plt.gcf().canvas.manager.set_window_title(name)
    plt.show()


def order_points(pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas

    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]


    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates

    return rect


def four_point_transform(image, pts, scale_factor=5):
    # obtain a consistent order of the points and unpack them
    # individually
    #img_temp = image.copy()
    #cv2.polylines(img_temp, [pts], True, (0, 0, 255))
    #print(pts)
    #show_image('pts', img_temp)

    #rect = order_points(pts)
    rect = pts.astype(np.float32)

    (tl, tr, br, bl) = rect


    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) +
                     ((br[1] - bl[1]) ** 2))*scale_factor
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) +
                     ((tr[1] - tl[1]) ** 2))*scale_factor
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) +
                      ((tr[1] - br[1]) ** 2))*scale_factor
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) +
                      ((tl[1] - bl[1]) ** 2))*scale_factor
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1],
         [0, 0],
        [maxWidth - 1, 0]],
        dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped
