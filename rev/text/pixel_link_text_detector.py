import tensorflow as tf
import tf_slim as slim
from tensorflow.python.ops import array_ops
from . import pixel_link_net
import numpy as np
#import cv2
from cv2 import cv2
import os

r_mean = 123.
g_mean = 117.
b_mean = 104.

pixel_conf_threshold = 0.6
link_conf_threshold = 0.9
tf.compat.v1.disable_eager_execution()


CWD = os.path.dirname(os.path.abspath(__file__))
text_detection_model_path =  os.path.join(CWD, '..','..','models','pixel_link_text_detection','model.ckpt-46231')

def image_whitened(image):
    means=[r_mean, g_mean, b_mean]
    mean = tf.constant(means, dtype=image.dtype)
    image = image - mean
    return image


def image_dimensions(image):
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(3).as_list()
        dynamic_shape = array_ops.unstack(array_ops.shape(image), 3)
        return [s if s is not None else d for s, d in zip(static_shape, dynamic_shape)]

def resize_image(image, size, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):
    with tf.compat.v1.name_scope('resize_image'):
        height, width, channels = image_dimensions(image)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize(image, size, method, align_corners)
        image = tf.reshape(image, tf.stack([size[0], size[1], channels]))
        return image


def preprocess_image(image, out_shape, data_format='NHWC'):
    with tf.compat.v1.name_scope('ssd_preprocessing_train'):
        image = tf.cast(image, dtype=tf.float32)
        image = image_whitened(image)
        image = resize_image(image, out_shape)
    
        if data_format == 'NCHW':
            image = tf.transpose(a=image, perm=(2, 0, 1))
    
    return image


def get_neighbours_8(x, y):
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), (x - 1, y), (x + 1, y), (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]

def is_valid_cord(x, y, w, h):
    return x >=0 and x < w and y >= 0 and y < h


def find_parent(point, group_mask):
    return group_mask[point]
        
def set_parent(point, parent, group_mask):
    group_mask[point] = parent
        
def is_root(point, group_mask):
    return find_parent(point, group_mask) == -1

def find_root(point, group_mask):
    root = point
    update_parent = False
    while not is_root(root, group_mask):
        root = find_parent(root, group_mask)
        update_parent = True
    
    # for acceleration of find_root
    if update_parent:
        set_parent(point, root, group_mask)
        
    return root

def join(p1, p2, group_mask):
    root1 = find_root(p1, group_mask)
    root2 = find_root(p2, group_mask)
    
    if root1 != root2:
        set_parent(root1, root2, group_mask)


def get_index(root, root_map):
    if root not in root_map:
        root_map[root] = len(root_map) + 1
    return root_map[root]

def get_all(pixel_mask, group_mask, points):
    root_map = {}
    mask = np.zeros_like(pixel_mask, dtype = np.int32)
    for point in points:
        point_root = find_root(point, group_mask)
        bbox_idx = get_index(point_root, root_map)
        mask[point] = bbox_idx
    
    return mask


def decode_image_by_join(pixel_scores, link_scores, pixel_conf_threshold, link_conf_threshold):
    pixel_mask = pixel_scores >= pixel_conf_threshold
    link_mask = link_scores >= link_conf_threshold
    points = list(zip(*np.where(pixel_mask)))
    h, w = np.shape(pixel_mask)
    group_mask = dict.fromkeys(points, -1)

    for point in points:
        y, x = point
        neighbours = get_neighbours_8(x, y)
        for n_idx, (nx, ny) in enumerate(neighbours):
            if is_valid_cord(nx, ny, w, h):
                link_value = link_mask[y, x, n_idx]
                pixel_cls = pixel_mask[ny, nx]
                if link_value and pixel_cls:
                    join(point, (ny, nx), group_mask)
    
    mask = get_all(pixel_mask, group_mask, points)
    return mask

def decode_image(pixel_scores, link_scores, pixel_conf_threshold, link_conf_threshold):
    mask =  decode_image_by_join(pixel_scores, link_scores, pixel_conf_threshold, link_conf_threshold)
    return mask

def decode_batch(pixel_cls_scores, pixel_link_scores, mpixel_conf_threshold = None, mlink_conf_threshold = None):
    
    if mpixel_conf_threshold is None:
        mpixel_conf_threshold = pixel_conf_threshold
    
    if mlink_conf_threshold is None:
        mlink_conf_threshold = link_conf_threshold
    
    batch_size = pixel_cls_scores.shape[0]
    batch_mask = []
    for image_idx in range(batch_size):
        image_pos_pixel_scores = pixel_cls_scores[image_idx, :, :]
        image_pos_link_scores = pixel_link_scores[image_idx, :, :, :]    
        mask = decode_image(
            image_pos_pixel_scores, image_pos_link_scores, 
            pixel_conf_threshold, link_conf_threshold
        )
        batch_mask.append(mask)
    return np.asarray(batch_mask, np.int32)


def tf_decode_score_map_to_mask_in_batch(pixel_cls_scores, pixel_link_scores):
    masks = tf.compat.v1.py_func(decode_batch, [pixel_cls_scores, pixel_link_scores], tf.int32)
    b, h, w = pixel_cls_scores.shape.as_list()
    masks.set_shape([b, h, w])
    return masks


def imread(path, rgb = False, mode = None):
    if mode is None:
        mode = cv2.IMREAD_COLOR
    img = cv2.imread(path, mode)
    if img is None:
        raise IOError('File not found:%s'%(path))
        
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def cast(obj, dtype):
    if isinstance(obj, list):
        return np.asarray(obj, dtype = tf.keras.backend.floatx())
    return np.cast[dtype](obj)

def resize(img, size = None, f = None, fx = None, fy = None, interpolation = cv2.INTER_LINEAR):
    
    h, w = np.shape(img)[0:2]
    if fx != None and fy != None:
        return cv2.resize(img, None, fx = fx, fy = fy, interpolation = interpolation)
        
    if size != None:
        size = cast(size, 'int')
        size = tuple(size)
        return cv2.resize(img, size, interpolation = interpolation)
    
    return cv2.resize(img, None, fx = f, fy = f, interpolation = interpolation)


def find_contours(mask, method = cv2.CHAIN_APPROX_SIMPLE):



    mask = np.asarray(mask, dtype = np.uint8)
    mask = mask.copy()
    try:
        contours, _ = cv2.findContours(mask, mode = cv2.RETR_CCOMP, 
                                   method = method)
    except:
        _, contours, _ = cv2.findContours(mask, mode = cv2.RETR_CCOMP, 
                                  method = method)


    return contours


def min_area_rect(cnt):
    rect = cv2.minAreaRect(cnt)
    cx, cy = rect[0]
    w, h = rect[1]
    theta = rect[2]
    box = [cx, cy, w, h, theta]
    return box, w * h

def rect_to_xys(rect, image_shape):
    h, w = image_shape[0:2]
    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x
    
    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y
    
    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    for i_xy, (x, y) in enumerate(points):
        x = get_valid_x(x)
        y = get_valid_y(y)
        points[i_xy, :] = [x, y]
    points = np.reshape(points, -1)
    return points

def mask_to_bboxes(mask, image_shape, min_area = 300, min_height = 10):
    bboxes = []
    image_h, image_w = image_shape[0:2]
    max_bbox_idx = mask.max()
    mask = resize(img = mask, size = (image_w, image_h), interpolation = cv2.INTER_NEAREST)
    
    mask2 = np.where(mask == 0, 0, 255)
    img = np.zeros(image_shape)

    
    for bbox_idx in range(1, max_bbox_idx + 1):
        bbox_mask = mask == bbox_idx
        cnts = find_contours(bbox_mask)
        if len(cnts) == 0:
            continue
        cnt = cnts[0]
        rect, rect_area = min_area_rect(cnt)

        w, h = rect[2:-1]
        #print('joao', min(w, h), w, h)
        if min(w, h) < min_height:
            continue
        xys = rect_to_xys(rect, image_shape)
        bboxes.append(xys)
    return bboxes

class PixelLinkDetector:
    def __init__(self, path_model=text_detection_model_path, preproc_img_w = 1280, preproc_img_h = 768):
        self.path_model = path_model
        self.preproc_img_w = preproc_img_w
        self.preproc_img_h = preproc_img_h

        #need for use tf 2
        tf.compat.v1.disable_eager_execution()
    
    def init(self):
        tf.reset_default_graph()
        global_step = slim.get_or_create_global_step()

        with tf.compat.v1.name_scope('evaluation_%dx%d' % (self.preproc_img_h, self.preproc_img_w)):
            with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=False):
                self.image = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None, 3])
                self.image_shape = tf.compat.v1.placeholder(dtype=tf.int32, shape=[3, ])
                processed_image = preprocess_image(self.image, (self.preproc_img_h, self.preproc_img_w))
                b_image = tf.expand_dims(processed_image, axis=0)
                self.net = pixel_link_net.PixelLinkNet(b_image, is_training = False)
                self.masks = tf_decode_score_map_to_mask_in_batch(self.net.pixel_pos_scores, self.net.link_pos_scores)
        

        sess_config = tf.compat.v1.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        variable_averages = tf.train.ExponentialMovingAverage(0.9999)
        variables_to_restore = variable_averages.variables_to_restore( tf.compat.v1.trainable_variables())
        variables_to_restore[global_step.op.name] = global_step
        self.saver = tf.compat.v1.train.Saver(var_list=variables_to_restore)

        self.session = tf.compat.v1.Session()
        self.saver.restore(self.session, self.path_model)

    def predict(self, img_path, image_idx = 0):
        image_data = imread(img_path)
        h, w, _ = image_data.shape

        #filter
        scale_factor = 1.5
        image_data = cv2.resize(image_data,(int(w*scale_factor),int(h*scale_factor)), interpolation=cv2.INTER_AREA)
        image_data = cv2.GaussianBlur(image_data,(7,7),sigmaX = 5, sigmaY=1)

        #apply model
        link_scores, pixel_scores, mask_vals = self.session.run([self.net.link_pos_scores, self.net.pixel_pos_scores, self.masks], feed_dict={self.image: image_data})

        pixel_score = pixel_scores[image_idx, ...]
        mask = mask_vals[image_idx, ...]

        bboxes_det = mask_to_bboxes(mask, image_data.shape)

        #recovery original coords
        bboxes_det_aux = []
        for bbox in bboxes_det:
            bbox = bbox/scale_factor
            bbox = bbox.astype(int)
            bboxes_det_aux.append(bbox)
        bboxes_det = bboxes_det_aux

        return bboxes_det

    def predict_multiple(self, img_paths):

        lsboxes = []
        for i, img_path in enumerate(img_paths):
            lsboxes.append( self.predict(img_path, i) )

        return lsboxes



def text_detect(path_image, path_model = text_detection_model_path, img_width = 1280, img_height = 768):

    tf.reset_default_graph()

    global_step = slim.get_or_create_global_step()

    with tf.compat.v1.name_scope('evaluation_%dx%d' % (img_height, img_width)):
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=False):
            image = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None, 3])
            image_shape = tf.compat.v1.placeholder(dtype=tf.int32, shape=[3, ])

            processed_image = preprocess_image(image, (img_height, img_width))

            b_image = tf.expand_dims(processed_image, axis=0)

            net = pixel_link_net.PixelLinkNet(b_image, is_training = False)

            masks = tf_decode_score_map_to_mask_in_batch(net.pixel_pos_scores, net.link_pos_scores)

            #print('mask type: ', type( masks ) )


    sess_config = tf.compat.v1.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    #sess_config.gpu_options.allow_growth = True
    variable_averages = tf.train.ExponentialMovingAverage(0.9999)
    variables_to_restore = variable_averages.variables_to_restore( tf.compat.v1.trainable_variables())
    variables_to_restore[global_step.op.name] = global_step
    saver = tf.compat.v1.train.Saver(var_list=variables_to_restore)

    with tf.compat.v1.Session() as sess:

        saver.restore(sess, path_model)

        #####

        image_data = imread(path_image)
        h, w, _ = image_data.shape

        #filter
        scale_factor = 1.5
        image_data = cv2.resize(image_data,(int(w*scale_factor),int(h*scale_factor)), interpolation=cv2.INTER_AREA)
        image_data = cv2.GaussianBlur(image_data,(7,7),sigmaX = 5, sigmaY=1)

        link_scores, pixel_scores, mask_vals = sess.run([net.link_pos_scores, net.pixel_pos_scores, masks], feed_dict={image: image_data})

        image_idx = 0
        pixel_score = pixel_scores[image_idx, ...]
        mask = mask_vals[image_idx, ...]

        bboxes_det = mask_to_bboxes(mask, image_data.shape)

        #recovery original coords
        bboxes_det_aux = []
        for bbox in bboxes_det:
            bbox = bbox/scale_factor
            bbox = bbox.astype(int)
            bboxes_det_aux.append(bbox)
        bboxes_det = bboxes_det_aux

    return bboxes_det