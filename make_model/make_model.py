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
        xmin.append(float(object_annotations['bbox']['xmin']))
        xmax.append(float(object_annotations['bbox']['xmax']))
        ymin.append(float(object_annotations['bbox']['ymin']))
        ymax.append(float(object_annotations['bbox']['ymax']))
        category_ids.append(int(object_annotations['name']))
        attributes.append(tf.train.Feature(
            int64_list=tf.train.Int64List(value=object_annotations['attributes'])))
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
        #     tf.train.FeatureList(feature=attributes)
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
            attr = []
            if (obj['attributes'] is not None):
                obj['attributes'] = obj['attributes'].split(',')
                for attribute in obj['attributes']:
                    if(labels.__contains__(attribute)):
                        attr.append(attributes.index(attribute) + 1)
                    else:
                        attributes.append(attribute)
                        attr.append(attributes.index(attribute) + 1)
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
            output_path, 'attributes.pbtxt'), 'wb') as fid:
        fid.write("""
# Embedded SSD with Mobilenet v1 configuration for MSCOCO Dataset.
# Users should configure the fine_tune_checkpoint field in the train config as
# well as the label_map_path and input_path fields in the train_input_reader and
# eval_input_reader. Search for "PATH_TO_BE_CONFIGURED" to find the fields that
# should be configured.

model {\
  ssd {\
    num_classes: """ + str(len(labels)) + "\\
    box_coder {\\
      faster_rcnn_box_coder {\
        y_scale: 10.0\
        x_scale: 10.0\
        height_scale: 5.0\
        width_scale: 5.0\
      }\
    }\
    matcher {\
      argmax_matcher {\
        matched_threshold: 0.5\
        unmatched_threshold: 0.5\
        ignore_thresholds: false\
        negatives_lower_than_unmatched: true\
        force_match_for_each_row: true\
      }\
    }\
    similarity_calculator {\
      iou_similarity {\
      }\
    }\
    anchor_generator {\
      ssd_anchor_generator {\
        num_layers: 5\
        min_scale: 0.2\
        max_scale: 0.95\
        aspect_ratios: 1.0\
        aspect_ratios: 2.0\
        aspect_ratios: 0.5\
        aspect_ratios: 3.0\
        aspect_ratios: 0.3333\
      }\
    }\
    image_resizer {\
      fixed_shape_resizer {\
        height: 256\
        width: 256\
      }\
    }\
    box_predictor {\
      convolutional_box_predictor {\
        min_depth: 0\
        max_depth: 0\
        num_layers_before_predictor: 0\
        use_dropout: false\
        dropout_keep_probability: 0.8\
        kernel_size: 1\
        box_code_size: 4\
        apply_sigmoid_to_scores: false\
        conv_hyperparams {\
          activation: RELU_6,\
          regularizer {\
            l2_regularizer {\
              weight: 0.00004\
            }\
          }\
          initializer {\
            truncated_normal_initializer {\
              stddev: 0.03\
              mean: 0.0\
            }\
          }\
          batch_norm {\
            train: true,\
            scale: true,\
            center: true,\
            decay: 0.9997,\
            epsilon: 0.001,\
          }\
        }\
      }\
    }\
    feature_extractor {\
      type: 'embedded_ssd_mobilenet_v1'\
      min_depth: 16\
      depth_multiplier: 0.125\
      conv_hyperparams {\
        activation: RELU_6,\
        regularizer {\
          l2_regularizer {\
            weight: 0.00004\
          }\
        }\
        initializer {\
          truncated_normal_initializer {\
            stddev: 0.03\
            mean: 0.0\
          }\
        }\
        batch_norm {\
          train: true,\
          scale: true,\
          center: true,\
          decay: 0.9997,\
          epsilon: 0.001,\
        }\
      }\
    }\
    loss {\
      classification_loss {\
        weighted_sigmoid {\
        }\
      }\
      localization_loss {\
        weighted_smooth_l1 {\
        }\
      }\
      hard_example_miner {\
        num_hard_examples: 3000\
        iou_threshold: 0.99\
        loss_type: CLASSIFICATION\
        max_negatives_per_positive: 3\
        min_negatives_per_image: 0\
      }\
      classification_weight: 1.0\
      localization_weight: 1.0\
    }\
    normalize_loss_by_num_matches: true\
    post_processing {\
      batch_non_max_suppression {\
        score_threshold: 1e-8\
        iou_threshold: 0.6\
        max_detections_per_class: 100\
        max_total_detections: 100\
      }\
      score_converter: SIGMOID\
    }\
  }\
}\
\
train_config: {\
  batch_size: 32\
  optimizer {\
    rms_prop_optimizer: {\
      learning_rate: {\
        exponential_decay_learning_rate {\
          initial_learning_rate: 0.004\
          decay_steps: 800720\
          decay_factor: 0.95\
        }\
      }\
      momentum_optimizer_value: 0.9\
      decay: 0.9\
      epsilon: 1.0\
    }\
  }\
  fine_tune_checkpoint: "/PATH_TO_BE_CONFIGU\RED/model.ckpt"
  data_augmentation_options {\
    random_horizontal_flip {\
    }\
  }\
  data_augmentation_options {\
    ssd_random_crop {\
    }\
  }\
}\
\
train_input_reader: {\
  tf_record_input_reader {\
    input_path: """ + '"' +  + '"' + """\
  }\
  label_map_path: """+ '"' +  + '"' + """\
}\
\
eval_config: {\
  num_examples: 8000\
  use_moving_averages: true\
}\
\
eval_input_reader: {\
  tf_record_input_reader {\
    input_path: """+ '"' +  + '"' + """\
  }\
  label_map_path: """+ '"' +  + '"' + """\
  shuffle: false\
  num_readers: 1\
}\
        """)\
    return\
\
\
def main(_):\
    # assert FLAGS.train_image_dir, '`train_\image_dir` missing.'
    # assert FLAGS.val_image_dir, '`val_imag\e_dir` missing.'
    # assert FLAGS.test_image_dir, '`test_im\age_dir` missing.'
    # assert FLAGS.train_annotations_file, '\`train_annotations_file` missing.'
    # assert FLAGS.val_annotations_file, '`v\al_annotations_file` missing.'
    # assert FLAGS.testdev_annotations_file,\ '`testdev_annotations_file` missing.'
    # labels, attributes, images = readAnnot\ations('/home/elias/Desktop/web/morvision/LabelMeAnnotationTool/Annotations/robots',
    #                 '/home/elias/Desktop/w\eb/morvision/LabelMeAnnotationTool/Images/robots',
    #                 None)\
    # print(images)\
\
    annotationsToExamples('/home/elias/Deskt\op/web/morvision/LabelMeAnnotationTool/Annotations/robots',
                          '/home/elias/Desktop/web/morvision/LabelMeAnnotationTool/Images/robots',
                          '/home/elias/Desktop/web/morvision/example/',
                          None)


if __name__ == '__main__':
    tf.app.run()
