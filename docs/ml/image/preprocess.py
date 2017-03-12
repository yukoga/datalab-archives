# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example dataflow pipeline for preparing image training data.


The output data is in format accepted by Cloud ML framework.

This tool expects as input a Google Cloud Storage directory with images
sorted into subdirectories based on their labels.

This tool produces following outputs:
input - URI to csv file, using format:
gs://image_uri1,labela,labelb,labelc
gs://image_uri2,labela,labeld
...

input_dict - URI to text file listing all labels using format
labela
labelb
labelc

It then creates one training example per each line of the created csv file.
When processing CSV file:
- all labels that are not present in input_dict are skipped
- all lines that have no label are skipped

To execute this pipeline on the cloud using default options, run this script with no arguments.

To execute this pipeline locally, specify:
  --runner=DirectPipelineRunner
  --output=gs://YOUR_OUTPUT_PREFIX

To execute this pipeline on the cloud using the Dataflow service and non-default options, specify
the pipeline configuration on the command line:
  --output gs://YOUR_OUTPUT_PREFIX
  --job_name NAME_FOR_YOUR_JOB
  --project YOUR_PROJECT_NAME
  --staging_location gs://YOUR_STAGING_DIRECTORY
  --temp_location gs://YOUR_TEMPORARY_DIRECTORY
  --runner BlockingDataflowPipelineRunner
  --num_workers NUM_WORKERS_TO_USE
  --disk_size_gb PER_WORKER_DISK_SIZE_GB
"""

import cStringIO
import csv
import datetime
import hashlib
import logging
import math
import os
import re
import subprocess
import sys

from PIL import Image
import tensorflow as tf

import google.cloud.dataflow as df
from google.cloud.dataflow.io import gcsio
from google.cloud.dataflow.utils.options import PipelineOptions
from google.protobuf import json_format

class Dataset(object):
  """Names of supported datasets."""
  TRAIN = 'train'
  TEST = 'test'
  ALL = [TRAIN, TEST]


class Default(object):
  """Default values of variables."""
  IMAGE_TYPE = 'jpeg'
  IMAGE_MAX_WIDTH = 640
  IMAGE_MAX_HEIGHT = 480
  IMAGE_GRAPH_FILENAME = 'classify_image_graph_def.pb'
  IMAGE_GRAPH_JPEG_INPUT_TENSOR = 'DecodeJpeg/contents:0'
  IMAGE_GRAPH_OUTPUT_TENSOR = 'pool_3/_reshape:0'
  DATA_LOCATION = 'gs://cloud-ml-data/img/flower_photos'


class StageName(object):
  """Names of dataflow pipeline stages."""
  READ_DICTIONARY = 'Read dict'
  READ_CSV = 'Read csv'
  PARSE_CSV = 'Parse csv'
  EXTRACT_LABEL_IDS = 'Extract label ids'
  READ_IMAGE = 'Read image'
  CONVERT_IMAGE = 'Convert image'
  ENCODE_EXAMPLE = 'Encode example'
  FILTER = 'Filter'
  TO_JSON = 'ToJson'
  SAVE = 'Save'


class PrepareImagesOptions(PipelineOptions):

  @classmethod
  def _add_argparse_args(cls, parser):
    parser.add_argument(
        '--input_data_location',
        dest='input_data_location',
        default=Default.DATA_LOCATION,
        help='Input directory containing images in labelled subdirectories.')
    parser.add_argument(
        '--input',
        dest='input',
        required=False,
        help='Input specified as uri to CSV file. Each line of csv file '
             'contains colon-separated GCS uri to an image and labels.')
    parser.add_argument(
        '--input_dict',
        dest='input_dict',
        required=False,
        help='Input dictionary. Specified as text file uri. '
             'Each line of the file stores one label.')
    parser.add_argument(
        '--output',
        dest='output',
        required=True,
        help='Output directory to write results to.')
    parser.add_argument(
        '--output_shard_count',
        dest='output_shard_count',
        default=10,
        help='Number of shards that should be used to store output files.')
    parser.add_argument(
        '--image_graph_jpeg_input_tensor',
        dest='image_graph_jpeg_input_tensor',
        default=Default.IMAGE_GRAPH_JPEG_INPUT_TENSOR,
        help='Name of tensor that accepts image content in jpeg format.')
    parser.add_argument(
        '--image_graph_output_tensor',
        dest='image_graph_output_tensor',
        default=Default.IMAGE_GRAPH_OUTPUT_TENSOR,
        help='Name of tensor which output should be stored as "embedding".')
    parser.add_argument(
        '--max_image_width',
        dest='max_image_width',
        default=Default.IMAGE_MAX_WIDTH,
        help='Maximum width of an image. If image width is larger '
             'the image will be scaled down.')
    parser.add_argument(
        '--max_image_height',
        dest='max_image_height',
        default=Default.IMAGE_MAX_HEIGHT,
        help='Maximum height of an image. If image height is larger '
             'the image will be scaled down.')
    parser.add_argument(
        '--training_data_percentage',
        dest='training_data_percentage',
        default=90,
        help='Percentage of examples to be used for training. '
        'Rest is used for testing. Value in range [0-100]')


def _open(uri, mode='rb'):
  if uri.startswith('gs://'):
    return gcsio.GcsIO().open(uri, mode)
  else:
    return open(uri, mode)


class ExtractLabelIdsDoFn(df.DoFn):
  """Encodes row into (uri, label_ids).
  """

  def start_bundle(self, context, all_labels, **kwargs):
    self.label_encoder = {}
    for i, label in enumerate(all_labels):
      self.label_encoder[label.strip()] = i

  def process(self, context, all_labels):
    # Row format is:
    # image_uri(,label_ids)*
    row = context.element

    label_ids = []
    for i in range(1, len(row)):
      for label in row[i].split(','):
        if label in self.label_encoder:
          label_ids.append(int(self.label_encoder[label.strip()]))

    yield row[0], label_ids


class ExtractImageDoFn(df.DoFn):
  """Encodes (uri, label_ids) into (uri, label_ids, image_bytes).
  """

  def process(self, context):
    uri, label_ids = context.element

    with _open(uri) as f:
      yield uri, label_ids, f.read()


class ResizeImageDoFn(df.DoFn):
  """Processes (uri, label_ids, image_bytes) to given format and size.

  Attributes:
    image_format: a string describing output image format.
    max_image_width: max image width, if input image is larger it is re-sized.
    max_image_height: max image height, if input image is larger it is
                      re-sized.
  """

  def __init__(self, image_format, max_image_width, max_image_height):
    self.image_format = image_format
    self.max_width = max_image_width
    self.max_height = max_image_height

  def process(self, context):
    def _maybe_scale_down(width, height, max_image_width, max_image_height):
      width_ratio = float(width) / float(max_image_width)
      height_ratio = float(height) / float(max_image_height)
      if height_ratio <= 1.0 and width_ratio <= 1.0:
        return (width, height)
      if height_ratio > width_ratio:
        # Use height based scaling.
        return (int(math.floor(float(width) / height_ratio)), max_image_height)
      # Use width based scaling.
      return (max_image_width, int(math.floor(float(height) / width_ratio)))

    uri, label_ids, image_bytes = context.element

    img = Image.open(cStringIO.StringIO(image_bytes))
    if img.size[0] > self.max_width or img.size[1] > self.max_height:
      img = img.resize(
          _maybe_scale_down(
              img.size[0], img.size[1], self.max_width,
              self.max_height))

    # Convert to desired format and output.
    output = cStringIO.StringIO()
    img.save(output, self.image_format)
    image_bytes = output.getvalue()
    yield uri, label_ids, image_bytes


class EncodeExampleDoFn(df.DoFn):
  """Encodes (uri, label_ids, image_bytes) into (tensorflow.Example, Dataset).

  Output proto contains 'label', 'image_uri' and 'embedding'.
  The 'embedding' is calculated by feeding image into input layer of image
  neural network and reading output of the bottleneck layer of the network.

  Attributes:
    image_graph_uri: an uri to gcs bucket where serialized image graph is
                     stored.
    image_graph_jpeg_input_tensor: name of input tensor that accepts jpeg data.
    image_graph_output_tensor: name of output tensor that produces embedding.
    training_data_percentage: percentage of data that should be used for
                              training.
  """

  def __init__(self, image_graph_uri, image_graph_jpeg_input_tensor,
               image_graph_output_tensor, training_data_percentage):
    self.image_graph_uri = image_graph_uri
    self.image_graph_jpeg_input_tensor = image_graph_jpeg_input_tensor
    self.image_graph_output_tensor = image_graph_output_tensor
    self.train_percent = training_data_percentage
    self.tf_session = None

  def start_bundle(self, context):
    # There is one tensorflow session per instance of EncodeExampleDoFn.
    # Image model graph is loaded and cached inside of the session.
    # The same instance of session is re-used between bundles.
    # Session is closed by destructor of Session object, which is called when
    # instance of EncodeExampleDoFn() id destructed.
    if not self.tf_session:
      self.tf_session = tf.InteractiveSession()
      with _open(self.image_graph_uri) as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

  def process(self, context):
    def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    uri, label_ids, image_bytes = context.element

    embedding = self.tf_session.run(
        self.tf_session.graph.get_tensor_by_name(
            self.image_graph_output_tensor),
        feed_dict={
            self.image_graph_jpeg_input_tensor: image_bytes,
        })
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_uri': _bytes_feature([uri]),
        'embedding': _float_feature(embedding.ravel().tolist())
    }))
    if label_ids:
      example.features.feature['label'].int64_list.value.extend(label_ids)

    fp = 100.0 * int(hashlib.sha1(uri).hexdigest()[0:4], 16) / int('ffff', 16)
    dataset = Dataset.TRAIN if fp < self.train_percent else Dataset.TEST
    yield example, dataset

class ImagePreprocessor(object):
  """Runs the pre-processing pipeline.
  """

  def __init__(self, args):
    self.pipeline_options = PipelineOptions(args)

  def preprocess(self, input_path, input_dict, output_path):
    """

    Args:
      input_path: Input specified as uri to CSV file. Each line of csv file
                  contains colon-separated GCS uri to an image and labels
      input_dict: Input dictionary. Specified as text file uri.
                  Each line of the file stores one label.
    """
    opt = self.pipeline_options.view_as(PrepareImagesOptions)
    p = df.Pipeline(options=self.pipeline_options)

    # Read input data.
    csv_data = df.io.TextFileSource(input_path, strip_trailing_newlines=True)
    dict_data = df.io.TextFileSource(input_dict, strip_trailing_newlines=True)
    labels = (p | df.Read(StageName.READ_DICTIONARY, dict_data))
    content = (p | df.Read(StageName.READ_CSV, csv_data)
               | df.Map(StageName.PARSE_CSV,
                        lambda line: csv.reader([line]).next())
               | df.ParDo(StageName.EXTRACT_LABEL_IDS, ExtractLabelIdsDoFn(),
                          df.pvalue.AsIter(labels))
               | df.ParDo(StageName.READ_IMAGE, ExtractImageDoFn()))

    # Process input data using common transformations.
    image_graph_uri = os.path.join(opt.input_data_location, Default.IMAGE_GRAPH_FILENAME)
    examples = (content
                | df.ParDo(StageName.CONVERT_IMAGE,
                           ResizeImageDoFn(Default.IMAGE_TYPE,
                                           opt.max_image_width,
                                           opt.max_image_height))
                | df.ParDo(StageName.ENCODE_EXAMPLE,
                           EncodeExampleDoFn(image_graph_uri,
                                             opt.image_graph_jpeg_input_tensor,
                                             opt.image_graph_output_tensor,
                                             opt.training_data_percentage)))

    # Write in JSON format to Text file.
    # Remove redundant whitespace for more compact representation.
    # Images/labels are base64 encoded so will not contain spaces.
    to_json = lambda x: re.sub(r'\s+', ' ', json_format.MessageToJson(x[0]))

    for dataset in Dataset.ALL:
      _ = (examples
           | df.Filter(StageName.FILTER + dataset,
                       lambda x, dataset=dataset: x[1] == dataset)
           | df.Map(StageName.TO_JSON + dataset, to_json)
           | df.Write(StageName.SAVE + dataset, df.io.TextFileSink(
               '{}.{}.json'.format(output_path, dataset),
               num_shards=opt.output_shard_count)))

    # Execute the pipeline.
    p.run()


class LocalImageTransformer(object):
  """Processes single image into tf.Example.

  Attributes:
    image_graph_uri: an uri to gcs bucket where serialized image graph is
                     stored.
    image_graph_jpeg_input_tensor: name of input tensor that accepts jpeg data.
    image_graph_output_tensor: name of output tensor that produces embedding.
    max_image_width: max image width, if input image is larger it is re-sized.
    max_image_height: max image height, if input image is larger it is
                      re-sized.

  Example call path:
    p = LocalImageTransformer(image_graph_uri='/tmp/classify_image_graph_def.pb')
    image_bytes = []
    uris = []
    for uri in ['/tmp/i1.jpg', '/tmp/i2.jpg', '/tmp/i3.jpg']:
      with _open(uri) as f:
        image_bytes.append(f.read())
        uris.append(uri)
        # Process single image at a time.
        tf_example = p.preprocess_list([image_bytes[-1]], [uris[-1]])

    # Alternatively process images in batches.
    tf_examples = p.preprocess_list(image_bytes, uris)
  """

  def __init__(
      self,
      input_data_location,
      image_graph_jpeg_input_tensor=Default.IMAGE_GRAPH_JPEG_INPUT_TENSOR,
      image_graph_output_tensor=Default.IMAGE_GRAPH_OUTPUT_TENSOR,
      max_image_width=Default.IMAGE_MAX_WIDTH,
      max_image_height=Default.IMAGE_MAX_HEIGHT):
    image_graph_uri = os.path.join(input_data_location, Default.IMAGE_GRAPH_FILENAME)
    self.resize_fn = ResizeImageDoFn(Default.IMAGE_TYPE,
                                     max_image_width,
                                     max_image_height)
    self.encode_fn = EncodeExampleDoFn(image_graph_uri,
                                       image_graph_jpeg_input_tensor,
                                       image_graph_output_tensor, 0)
    self.encode_fn.start_bundle(None)

  def callDoFn(self, func, element):
    class DoFnInput(object):
      def __init__(self, element):
        self.element = element
    context = DoFnInput(element)
    return func.process(context)

  def preprocess_list(self, image_bytes, uri=[]):
    """Converts batches of image_bytes into tf.Example protos."""
    data = [(u if u else 'uri', [], i) for i, u in map(None, image_bytes, uri)]
    data = [self.callDoFn(self.resize_fn, d) for d in data]
    data = [self.callDoFn(self.encode_fn, next(d)) for d in data]
    return [json_format.MessageToJson(next(d)[0]) for d in data]


def run(arg=None):
  preprocessor = ImagePreprocessor(args)
  opt = preprocessor.pipeline_options.view_as(PrepareImagesOptions)

  all_data_path = os.path.join(opt.output, 'all_data.csv')
  label_path = os.path.join(opt.output, 'dict.txt')
  output_path = os.path.join(opt.output, 'data', 'data')

  if not opt.input:
    create_inputs(opt.input_data_location, all_data_path, label_path)
  else:
    if not opt.input_dict:
      raise ValueError("--input_dict must be specified if --input is specified.")
    all_data_path = opt.input
    label_path = opt.input_dict
  preprocessed_datasets = preprocessor.preprocess(all_data_path, label_path, output_path)

def create_inputs(input_data_location, all_data_path, label_path):
  """Create csv input data for model.

  This function prepares csv data for training custom image model.
  It assumes that image files are organized into folders, and
  each folder is named with the label that should be attached to each
  image stored within the folder.
  """
  labels = set()
  image_list = subprocess.Popen(['gsutil', 'ls', os.path.join(input_data_location, '*/*.*')],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  all_data_content = ''
  for line in image_list.stdout:
    line = line.rstrip()
    paths = line.split("/")
    if len(paths) >= 2:
      labels.add(paths[-2])
      all_data_content += ("%s,%s\n" % (line, paths[-2]))

  with _open(all_data_path, 'w') as all_data:
    all_data.write(all_data_content)

  with _open(label_path, 'w') as label_dict:
    for label in labels:
      label_dict.write("%s\n" % (label))

  return all_data_path, label_path

def get_defaults():
  import google.cloud.ml as ml
  default_project = ml.Client.default().project_id
  output_dir = ml.Samples.get_bucket('image_classification')
  # Delete the ml module so that it doesn't get pickled up with our Dataflow code.
  del ml

  return default_project, output_dir

if __name__ == '__main__':
  if len(sys.argv) <= 1:
    job_name = 'cloud-ml-sdk-sample-image' + '-' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    default_project, output_dir = get_defaults()
    args = [
        '--output=' + output_dir,
        '--project=' + default_project,
        '--job_name=' + job_name,
        '--staging_location=' + os.path.join(os.path.join(output_dir, 'tmp'), 'staging'),
        '--temp_location=' + os.path.join(os.path.join(output_dir, 'tmp'), 'temp'),
        '--teardown_policy=TEARDOWN_ALWAYS',
        '--num_workers=10',
        '--disk_size_gb=50',
        '--runner=BlockingDataflowPipelineRunner'
    ]
  else:
    args = sys.argv
  run(args)

