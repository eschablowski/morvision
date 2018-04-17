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


def create_tf_example(image, include_masks):
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
    image_height = int(image['height'])
    image_width = int(image['width'])
    filename = image['file_name']
    image_id = image['id']

    key = hashlib.sha256(image['encoded']).hexdigest()

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    category_ids = []
    encoded_mask_png = []
    attributes = []
    for object_annotations in image['annotations']:
        attributes.append(object_annotations['attributes'])
        xmin.append(float(object_annotations['bbox']['xmin']))
        xmax.append(float(object_annotations['bbox']['xmax']))
        ymin.append(float(object_annotations['bbox']['ymin']))
        ymax.append(float(object_annotations['bbox']['ymax']))
        category_ids.append(int(object_annotations['name']))
        if include_masks:
            output_io = io.BytesIO(object_annotations['mask'])
            encoded_mask_png.append(output_io.getvalue())
    feature_dict = {
        'image/height':
            dataset_util.int64_feature(int(image_height)),
        'image/width':
            dataset_util.int64_feature(int(image_width)),
        'image/filename':
            dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':
            dataset_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded':
            dataset_util.bytes_feature(image['encoded']),
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
        # 'image/object/class/attributes':
        #     dataset_util.int64_list_feature(category_ids)
    }
    if include_masks:
        feature_dict['image/object/mask'] = (
            dataset_util.bytes_list_feature(encoded_mask_png))
    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))
    return example


def readAnnotations(
        annotations_dir, image_dir, masks_dir):
    annotations = [f for f in os.listdir(
        annotations_dir) if os.path.isfile(os.path.join(annotations_dir, f))]
    images = []
    labels = []
    attributes = []
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
                obj['name'] = data['object'][int(
                    obj['parts']['ispartof'])]['name'] + '/' + obj['name']
            if(labels.__contains__(obj['name'])):
                id = labels.index(obj['name']) + 1
            else:
                labels.append(obj['name'])
                id = labels.index(obj['name']) + 1
            attr = 0
            if (obj['attributes'] is not None):
                obj['attributes'] = obj['attributes'].split(
                    ',')
                obj['attributes'].sort()
                obj['attributes'] = ','.join(iter(obj['attributes']))
                if(attributes.__contains__(obj['attributes'])):
                    attr = attributes.index(obj['attributes']) + 1
                else:
                    attributes.append(obj['attributes'])
                    attr = attributes.index(obj['attributes']) + 1
            ann = {
                'bbox': {
                    'xmin': float(float(obj['segm']['box']['xmin']) / float(data['imagesize']['ncols'])),
                    'xmax': float(float(obj['segm']['box']['xmax']) / float(data['imagesize']['ncols'])),
                    'ymin': float(float(obj['segm']['box']['ymin']) / float(data['imagesize']['nrows'])),
                    'ymax': float(float(obj['segm']['box']['ymax']) / float(data['imagesize']['nrows']))
                },
                'id': obj['id'],
                'name': id,
                'partOf': obj['parts']['ispartof'],
                'attributes': attr
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
            'id': labels.index(label) + 1
        })
    labels = label_map_util.create_category_index(l)
    a = []
    for attribute in attributes:
        a.append({
            'name': attribute,
            'id': attributes.index(attribute) + 1
        })
    attributes = label_map_util.create_category_index(a)
    return labels, attributes, images


def annotationsToExamples(annotations_dir, image_dir, output_path, masks_dir):
    labels, attributes, images = readAnnotations(
        annotations_dir, image_dir, masks_dir)
    examples = []
    id = 0
    for image in images:
        image['id'] = id
        examples.append(create_tf_example(image, masks_dir is not None))
        id += 1
    writer = tf.python_io.TFRecordWriter(
        os.path.join(output_path, 'output.record'))
    for example in examples:
        writer.write(example.SerializeToString())
    writer.close()
    label_map_util.save_label_map_dict(
        os.path.join(output_path, 'labelmap.pbtxt'), labels)
    label_map_util.save_label_map_dict(os.path.join(
        output_path, 'attributes.pbtxt'), attributes)
    with tf.gfile.GFile(os.path.join(
            output_path, 'pipeline.config'), 'wb') as fid:
        fid.write("\n\
# Embedded SSD with Mobilenet v1 configuration for MSCOCO Dataset.\n\
# Users should configure the fine_tune_checkpoint field in the train config as\n\
# well as the label_map_path and input_path fields in the train_input_reader and\n\
# eval_input_reader. Search for  to find the fields that\n\
# should be configured.\n\
\n\
model {\n\
  ssd {\n\
    num_classes: " + str(len(labels)) + "\n\
    box_coder {\n\
      faster_rcnn_box_coder {\n\
        y_scale: 10.0\n\
        x_scale: 10.0\n\
        height_scale: 5.0\n\
        width_scale: 5.0\n\
      }\n\
    }\n\
    matcher {\n\
      argmax_matcher {\n\
        matched_threshold: 0.5\n\
        unmatched_threshold: 0.5\n\
        ignore_thresholds: false\n\
        negatives_lower_than_unmatched: true\n\
        force_match_for_each_row: true\n\
      }\n\
    }\n\
    similarity_calculator {\n\
      iou_similarity {\n\
      }\n\
    }\n\
    anchor_generator {\n\
      ssd_anchor_generator {\n\
        num_layers: 5\n\
        min_scale: 0.2\n\
        max_scale: 0.95\n\
        aspect_ratios: 1.0\n\
        aspect_ratios: 2.0\n\
        aspect_ratios: 0.5\n\
        aspect_ratios: 3.0\n\
        aspect_ratios: 0.3333\n\
      }\n\
    }\n\
    image_resizer {\n\
      fixed_shape_resizer {\n\
        height: 256\n\
        width: 256\n\
      }\n\
    }\n\
    box_predictor {\n\
      convolutional_box_predictor {\n\
        min_depth: 0\n\
        max_depth: 0\n\
        num_layers_before_predictor: 0\n\
        use_dropout: false\n\
        dropout_keep_probability: 0.8\n\
        kernel_size: 1\n\
        box_code_size: 4\n\
        apply_sigmoid_to_scores: false\n\
        conv_hyperparams {\n\
          activation: RELU_6,\n\
          regularizer {\n\
            l2_regularizer {\n\
              weight: 0.00004\n\
            }\n\
          }\n\
          initializer {\n\
            truncated_normal_initializer {\n\
              stddev: 0.03\n\
              mean: 0.0\n\
            }\n\
          }\n\
          batch_norm {\n\
            train: true,\n\
            scale: true,\n\
            center: true,\n\
            decay: 0.9997,\n\
            epsilon: 0.001,\n\
          }\n\
        }\n\
      }\n\
    }\n\
    feature_extractor {\n\
      type: 'embedded_ssd_mobilenet_v1'\n\
      min_depth: 16\n\
      depth_multiplier: 0.125\n\
      conv_hyperparams {\n\
        activation: RELU_6,\n\
        regularizer {\n\
          l2_regularizer {\n\
            weight: 0.00004\n\
          }\n\
        }\n\
        initializer {\n\
          truncated_normal_initializer {\n\
            stddev: 0.03\n\
            mean: 0.0\n\
          }\n\
        }\n\
        batch_norm {\n\
          train: true,\n\
          scale: true,\n\
          center: true,\n\
          decay: 0.9997,\n\
          epsilon: 0.001,\n\
        }\n\
      }\n\
    }\n\
    loss {\n\
      classification_loss {\n\
        weighted_sigmoid {\n\
        }\n\
      }\n\
      localization_loss {\n\
        weighted_smooth_l1 {\n\
        }\n\
      }\n\
      hard_example_miner {\n\
        num_hard_examples: 3000\n\
        iou_threshold: 0.99\n\
        loss_type: CLASSIFICATION\n\
        max_negatives_per_positive: 3\n\
        min_negatives_per_image: 0\n\
      }\n\
      classification_weight: 1.0\n\
      localization_weight: 1.0\n\
    }\n\
    normalize_loss_by_num_matches: true\n\
    post_processing {\n\
      batch_non_max_suppression {\n\
        score_threshold: 1e-8\n\
        iou_threshold: 0.6\n\
        max_detections_per_class: 100\n\
        max_total_detections: 100\n\
      }\n\
      score_converter: SIGMOID\n\
    }\n\
  }\n\
}\n\
\n\
train_config: {\n\
  batch_size: 32\n\
  optimizer {\n\
    rms_prop_optimizer: {\n\
      learning_rate: {\n\
        exponential_decay_learning_rate {\n\
          initial_learning_rate: 0.004\n\
          decay_steps: 800720\n\
          decay_factor: 0.95\n\
        }\n\
      }\n\
      momentum_optimizer_value: 0.9\n\
      decay: 0.9\n\
      epsilon: 1.0\n\
    }\n\
  }\n\
#  fine_tune_checkpoint: \n\
  data_augmentation_options {\n\
    random_horizontal_flip {\n\
    }\n\
  }\n\
  data_augmentation_options {\n\
    ssd_random_crop {\n\
    }\n\
  }\n\
}\n\
\n\
train_input_reader: {\n\
  tf_record_input_reader {\n\
    input_path: " + '"' + os.path.join(output_path, 'output.record') + '"' + "\n\
  }\n\
  label_map_path: " + '"' + os.path.join(output_path, 'labelmap.pbtxt') + '"' + "\n\
}\n\
\n\
eval_config: {\n\
  num_examples: 8000\n\
  use_moving_averages: true\n\
}\n\
\n\
eval_input_reader: {\n\
  tf_record_input_reader {\n\
    input_path: " + '"' + os.path.join(output_path, 'output.record') + '"' + "\n\
  }\n\
  label_map_path: " + '"' + os.path.join(output_path, 'labelmap.pbtxt') + '"' + "\n\
  shuffle: false\n\
  num_readers: 1\n\
}\n\
    ")
    return


def main(_):
    # assert FLAGS.train_image_dir, '`train_\image_dir` missing.'
    # assert FLAGS.val_image_dir, '`val_imag\e_dir` missing.'
    # assert FLAGS.test_image_dir, '`test_im\age_dir` missing.'
    # assert FLAGS.train_annotations_file, '\`train_annotations_file` missing.'
    # assert FLAGS.val_annotations_file, '`v\al_annotations_file` missing.'
    # assert FLAGS.testdev_annotations_file,\ '`testdev_annotations_file` missing.'
    # labels, attributes, images = readAnnot\ations('/home/elias/Desktop/web/morvision/LabelMeAnnotationTool/Annotations/robots',
    #                 '/home/elias/Desktop/w\eb/morvision/LabelMeAnnotationTool/Images/robots',
    #                 None)
    # print(images)
    annotationsToExamples('/home/elias/Desktop/web/morvision/LabelMeAnnotationTool/Annotations/robots',
                          '/home/elias/Desktop/web/morvision/LabelMeAnnotationTool/Images/robots',
                          '/home/elias/Desktop/web/morvision/example/',
                          None)


if __name__ == '__main__':
    tf.app.run()
