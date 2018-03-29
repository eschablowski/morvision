# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw COCO dataset to TFRecord for object_detection.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
from lxml import etree
import os
import numpy as np
import PIL.Image


from pycocotools import mask
import tensorflow as tf

from utils import dataset_util
from utils import label_map_util


flags = tf.app.flags
tf.flags.DEFINE_boolean('include_masks', False,
                        'Whether to include instance segmentations masks '
                        '(PNG encoded) in the result. default: False.')
tf.flags.DEFINE_string('train_image_dir', '',
                       'Training image directory.')
tf.flags.DEFINE_string('val_image_dir', '',
                       'Validation image directory.')
tf.flags.DEFINE_string('test_image_dir', '',
                       'Test image directory.')
tf.flags.DEFINE_string('train_annotations_file', '',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_annotations_file', '',
                       'Validation annotations JSON file.')
tf.flags.DEFINE_string('testdev_annotations_file', '',
                       'Test-dev annotations JSON file.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def create_tf_example(image,
                      annotations_list,
                      image_dir,
                      category_index,
                      include_masks=False):
    """Converts image and annotations to a tf.Example proto.

    Args:
      image: dict with keys:
        [u'license', u'file_name', u'coco_url', u'height', u'width',
        u'date_captured', u'flickr_url', u'id']
      annotations_list:
        list of dicts with keys:
        [u'segmentation', u'area', u'iscrowd', u'image_id',
        u'bbox', u'category_id', u'id']
        Notice that bounding box coordinates in the official COCO dataset are
        given as [x, y, width, height] tuples using absolute coordinates where
        x, y represent the top-left (0-indexed) corner.  This function converts
        to the format expected by the Tensorflow Object Detection API (which is
        which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
        to image size).
      image_dir: directory containing the image files.
      category_index: a dict containing COCO category information keyed
        by the 'id' field of each category.  See the
        label_map_util.create_category_index function.
      include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.
    Returns:
      example: The converted tf.Example
      num_annotations_skipped: Number of (invalid) annotations that were ignored.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']

    full_path = os.path.join(image_dir, filename)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    encoded_mask_png = []
    num_annotations_skipped = 0
    for object_annotations in annotations_list:
        (x, y, width, height) = tuple(object_annotations['bbox'])
        if width <= 0 or height <= 0:
            num_annotations_skipped += 1
            continue
        if x + width > image_width or y + height > image_height:
            num_annotations_skipped += 1
            continue
        xmin.append(float(x) / image_width)
        xmax.append(float(x + width) / image_width)
        ymin.append(float(y) / image_height)
        ymax.append(float(y + height) / image_height)
        is_crowd.append(object_annotations['iscrowd'])
        category_id = int(object_annotations['category_id'])
        category_ids.append(category_id)
        category_names.append(
            category_index[category_id]['name'].encode('utf8'))

        if include_masks:
            run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                                image_height, image_width)
            binary_mask = mask.decode(run_len_encoding)
            if not object_annotations['iscrowd']:
                binary_mask = np.amax(binary_mask, axis=2)
            pil_image = PIL.Image.fromarray(binary_mask)
            output_io = io.BytesIO()
            pil_image.save(output_io, format='PNG')
            encoded_mask_png.append(output_io.getvalue())
    feature_dict = {
        'image/height':
            dataset_util.int64_feature(image_height),
        'image/width':
            dataset_util.int64_feature(image_width),
        'image/filename':
            dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':
            dataset_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
        'image/format':
            dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
        'image/object/class/label':
            dataset_util.int64_list_feature(category_ids),
        #'image/object/class/attributes':

    }
    if include_masks:
        feature_dict['image/object/mask'] = (
            dataset_util.bytes_list_feature(encoded_mask_png))
    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))
    return key, example, num_annotations_skipped


def readAnnotations(
        annotations_dir, image_dir, output_path, masks_dir):
    annotations = [f for f in os.listdir(
        annotations_dir) if os.path.isfile(os.path.join(annotations_dir, f))]
    images = []
    annotations_dict = []
    labels = []
    for annotation in annotations:
        with tf.gfile.GFile(os.path.join(annotations_dir, annotation), 'r') as fid:
            xml = etree.fromstring(fid.read())
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        with tf.gfile.GFile(os.path.join(image_dir, data['filename']), 'rb') as fid:
            encoded_jpg = fid.read()
        image = {
            'file_name': data['filename'],
            'height': data['imagesize']['nrows'],
            'width': data['imagesize']['ncols'],
            'encoded': encoded_jpg,
            'format': 'jpeg'.encode('utf8')
        }
        anns = []
        for obj in data['object']:
            id = 0
            if(obj['parts']['ispartof'] is not None):
                obj['name'] = data['object'][int(obj['parts']['ispartof'])]['name'] + '/' + obj['name']
            if(labels.__contains__(obj['name'])):
                id = labels.index(obj['name'])
            else:
                labels.append(obj['name'])
                labels.index(obj['name'])
            ann = {
                'bbox': {
                    'xmin': float(float(obj['segm']['box']['xmin']) / float(data['imagesize']['ncols'])),
                    'xmax': float(float(obj['segm']['box']['xmax']) / float(data['imagesize']['ncols'])),
                    'ymin': float(float(obj['segm']['box']['ymin']) / float(data['imagesize']['nrows'])),
                    'ymax': float(float(obj['segm']['box']['ymax']) / float(data['imagesize']['nrows']))
                },
                'id': obj['id'],
                'name': id,
                'partOf': obj['parts']['ispartof']
            }
            if(masks_dir is not None):
                with tf.gfile.GFile(os.path.join(masks_dir, obj['mask'], 'rb')) as fid:
                    mask = fid.read()
                    ann['mask'] = mask
            anns.append(ann)
        image['annotations'] = anns
        images.append(image)
    l = []
    for label in labels:
        l.append({
            'name': label,
            'id': labels.index(label)
        })
    labels = label_map_util.create_category_index(l)
    return labels, images


def main(_):
    # assert FLAGS.train_image_dir, '`train_image_dir` missing.'
    # assert FLAGS.val_image_dir, '`val_image_dir` missing.'
    # assert FLAGS.test_image_dir, '`test_image_dir` missing.'
    # assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
    # assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'
    # assert FLAGS.testdev_annotations_file, '`testdev_annotations_file` missing.'

    labels, images = readAnnotations('/home/elias/Desktop/web/morvision/LabelMeAnnotationTool/Annotations/robots',
                                                  '/home/elias/Desktop/web/morvision/LabelMeAnnotationTool/Images/robots', '/home/elias/Desktop/web/morvision/robots',
                                                  None)
    print('\n\n\n\n')
    print(labels)
    print(images)


if __name__ == '__main__':
    tf.app.run()
