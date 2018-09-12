# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition for inception v1 classification network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import inception_utils

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

# Added by Marcelo
BFP_out = tf.load_op_library('/home/marcelo/tensorflow/Scripts/BFP/lib/bfp_out.so')

def inception_v1_base(inputs,
                      final_endpoint='Mixed_5c',
                      scope='InceptionV1',
                      m_w=1,
                      e_w=8,
                      s_w=3,
		      offset=0,
                      alter=False,
                      debug=False,
                      scale=False):
  """Defines the Inception V1 base architecture.

  This architecture is defined in:
    Going deeper with convolutions
    Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    http://arxiv.org/pdf/1409.4842v1.pdf.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
      'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
      'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
    scope: Optional variable_scope.

  Returns:
    A dictionary from components of the network to the corresponding activation.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values.
  """
  
  # Added by Marcelo
  # This is a flag to accept or not my modifications
  m_w_layer = []
  e_w_layer = []
  s_e_layer = []
  ofs_layer = []
  for _ in range(58):
    m_w_layer.append(m_w)
  for _ in range(58):
    e_w_layer.append(e_w)
  for _ in range(58):
    s_e_layer.append(s_w)
  fac = 0.1
  for _ in range(58):
    ofs_layer.append(offset)

  end_points = {}
  with tf.variable_scope(scope, 'InceptionV1', [inputs]):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_initializer=trunc_normal(0.01)):
      with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                          stride=1, padding='SAME'):
        end_point = 'Conv2d_1a_7x7'	
        net = inputs
        if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[0], ShDepth=s_e_layer[0], MWidth=m_w_layer[0], EWidth=e_w_layer[0])
        if(debug): net = tf.Print(net, [net], "Image Input: \n", summarize=999999999) 
        net = slim.conv2d(net, 64, [7, 7], stride=2, scope=end_point)
        if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[0], ShDepth=s_e_layer[0], MWidth=m_w_layer[0], EWidth=e_w_layer[0])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool_2a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'Conv2d_2b_1x1'
        net = slim.conv2d(net, 64, [1, 1], scope=end_point)
        if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[0], ShDepth=s_e_layer[1], MWidth=m_w_layer[1], EWidth=e_w_layer[1])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'Conv2d_2c_3x3'
        net = slim.conv2d(net, 192, [3, 3], scope=end_point)
        if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[0], ShDepth=s_e_layer[2], MWidth=m_w_layer[2], EWidth=e_w_layer[2])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool_3a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[0], ShDepth=s_e_layer[2], MWidth=m_w_layer[2], EWidth=e_w_layer[2])
        if(debug): net = tf.Print(net, [net], "After Last Convolution first Branch of Inception: \n", summarize=999999999)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_3b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[0], ShDepth=s_e_layer[3], MWidth=m_w_layer[3], EWidth=e_w_layer[3])
            branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[0], ShDepth=s_e_layer[4], MWidth=m_w_layer[4], EWidth=e_w_layer[4])
            branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
            if(alter): branch_1 = BFP_out.bfp_out(branch_1, offset=ofs_layer[0], ShDepth=s_e_layer[5], MWidth=m_w_layer[5], EWidth=e_w_layer[5])
            branch_1 = slim.conv2d(branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[0], ShDepth=s_e_layer[6], MWidth=m_w_layer[6], EWidth=e_w_layer[6])
            branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
            if(alter): branch_2 = BFP_out.bfp_out(branch_2, offset=ofs_layer[0], ShDepth=s_e_layer[7], MWidth=m_w_layer[7], EWidth=e_w_layer[7])
            branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            if(alter): branch_3 = BFP_out.bfp_out(branch_3, offset=ofs_layer[0], ShDepth=s_e_layer[8], MWidth=m_w_layer[8], EWidth=e_w_layer[8])
            branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[0], ShDepth=s_e_layer[3], MWidth=m_w_layer[3], EWidth=e_w_layer[3])
        if(debug): net = tf.Print(net, [net], "After Last Convolution second Branch of Inception: \n", summarize=999999999)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_3c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[9], ShDepth=s_e_layer[9], MWidth=m_w_layer[9], EWidth=e_w_layer[9])
            branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[10], ShDepth=s_e_layer[10], MWidth=m_w_layer[10], EWidth=e_w_layer[10])
            branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
            if(alter): branch_1 = BFP_out.bfp_out(branch_1, offset=ofs_layer[11], ShDepth=s_e_layer[11], MWidth=m_w_layer[11], EWidth=e_w_layer[11])
            branch_1 = slim.conv2d(branch_1, 192, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[12], ShDepth=s_e_layer[12], MWidth=m_w_layer[12], EWidth=e_w_layer[12])
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            if(alter): branch_2 = BFP_out.bfp_out(branch_2, offset=ofs_layer[13], ShDepth=s_e_layer[13], MWidth=m_w_layer[13], EWidth=e_w_layer[13])
            branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            if(alter): branch_3 = BFP_out.bfp_out(branch_3, offset=ofs_layer[14], ShDepth=s_e_layer[14], MWidth=m_w_layer[14], EWidth=e_w_layer[14])
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[0], ShDepth=s_e_layer[3], MWidth=m_w_layer[3], EWidth=e_w_layer[3])
        if(debug): net = tf.Print(net, [net], "After Last Convolution third Branch of Inception: \n", summarize=999999999)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool_4a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[15], ShDepth=s_e_layer[15], MWidth=m_w_layer[15], EWidth=e_w_layer[15])
            branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[16], ShDepth=s_e_layer[16], MWidth=m_w_layer[16], EWidth=e_w_layer[16])
            branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
            if(alter): branch_1 = BFP_out.bfp_out(branch_1, offset=ofs_layer[17], ShDepth=s_e_layer[17], MWidth=m_w_layer[17], EWidth=e_w_layer[17])
            branch_1 = slim.conv2d(branch_1, 208, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[18], ShDepth=s_e_layer[18], MWidth=m_w_layer[18], EWidth=e_w_layer[18])
            branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
            if(alter): branch_2 = BFP_out.bfp_out(branch_2, offset=ofs_layer[19], ShDepth=s_e_layer[19], MWidth=m_w_layer[19], EWidth=e_w_layer[19])
            branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            if(alter): branch_3 = BFP_out.bfp_out(branch_3, offset=ofs_layer[20], ShDepth=s_e_layer[20], MWidth=m_w_layer[20], EWidth=e_w_layer[20])
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[0], ShDepth=s_e_layer[3], MWidth=m_w_layer[3], EWidth=e_w_layer[3])
        if(debug): net = tf.Print(net, [net], "After third Inception Block: \n", summarize=999999999) 
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[21], ShDepth=s_e_layer[21], MWidth=m_w_layer[21], EWidth=e_w_layer[21])
            branch_0 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[22], ShDepth=s_e_layer[22], MWidth=m_w_layer[22], EWidth=e_w_layer[22])
            branch_1 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
            if(alter): branch_1 = BFP_out.bfp_out(branch_1, offset=ofs_layer[23], ShDepth=s_e_layer[23], MWidth=m_w_layer[23], EWidth=e_w_layer[23])
            branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[24], ShDepth=s_e_layer[24], MWidth=m_w_layer[24], EWidth=e_w_layer[24])
            branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
            if(alter): branch_2 = BFP_out.bfp_out(branch_2, offset=ofs_layer[25], ShDepth=s_e_layer[25], MWidth=m_w_layer[25], EWidth=e_w_layer[25])
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            if(alter): branch_3 = BFP_out.bfp_out(branch_3, offset=ofs_layer[26], ShDepth=s_e_layer[26], MWidth=m_w_layer[26], EWidth=e_w_layer[26])
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[0], ShDepth=s_e_layer[3], MWidth=m_w_layer[3], EWidth=e_w_layer[3])
        end_points[end_point] = net
        if(debug): net = tf.Print(net, [net], "After fourth Inception Block: \n", summarize=999999999) 
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4d'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[27], ShDepth=s_e_layer[27], MWidth=m_w_layer[27], EWidth=e_w_layer[27])
            branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[28], ShDepth=s_e_layer[28], MWidth=m_w_layer[28], EWidth=e_w_layer[28])
            branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
            if(alter): branch_1 = BFP_out.bfp_out(branch_1, offset=ofs_layer[29], ShDepth=s_e_layer[29], MWidth=m_w_layer[29], EWidth=e_w_layer[29])
            branch_1 = slim.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[30], ShDepth=s_e_layer[30], MWidth=m_w_layer[30], EWidth=e_w_layer[30])
            branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
            if(alter): branch_2 = BFP_out.bfp_out(branch_2, offset=ofs_layer[31], ShDepth=s_e_layer[31], MWidth=m_w_layer[31], EWidth=e_w_layer[31])
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            if(alter): branch_3 = BFP_out.bfp_out(branch_3, offset=ofs_layer[32], ShDepth=s_e_layer[32], MWidth=m_w_layer[32], EWidth=e_w_layer[32])
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[0], ShDepth=s_e_layer[3], MWidth=m_w_layer[3], EWidth=e_w_layer[3])
        end_points[end_point] = net
        if(debug): net = tf.Print(net, [net], "After fifth Inception Block: \n", summarize=999999999) 
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4e'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[33], ShDepth=s_e_layer[33], MWidth=m_w_layer[33], EWidth=e_w_layer[33])
            branch_0 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[34], ShDepth=s_e_layer[34], MWidth=m_w_layer[34], EWidth=e_w_layer[34])
            branch_1 = slim.conv2d(net, 144, [1, 1], scope='Conv2d_0a_1x1')
            if(alter): branch_1 = BFP_out.bfp_out(branch_1, offset=ofs_layer[35], ShDepth=s_e_layer[35], MWidth=m_w_layer[35], EWidth=e_w_layer[35])
            branch_1 = slim.conv2d(branch_1, 288, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[36], ShDepth=s_e_layer[36], MWidth=m_w_layer[36], EWidth=e_w_layer[36])
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            if(alter): branch_2 = BFP_out.bfp_out(branch_2, offset=ofs_layer[37], ShDepth=s_e_layer[37], MWidth=m_w_layer[37], EWidth=e_w_layer[37])
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            if(alter): branch_3 = BFP_out.bfp_out(branch_3, offset=ofs_layer[38], ShDepth=s_e_layer[38], MWidth=m_w_layer[38], EWidth=e_w_layer[38])
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[0], ShDepth=s_e_layer[3], MWidth=m_w_layer[3], EWidth=e_w_layer[3])
        end_points[end_point] = net
        if(debug): net = tf.Print(net, [net], "After sixth Inception Block: \n", summarize=999999999) 
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4f'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[40], ShDepth=s_e_layer[39], MWidth=m_w_layer[39], EWidth=e_w_layer[39])
            branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[41], ShDepth=s_e_layer[40], MWidth=m_w_layer[40], EWidth=e_w_layer[40])
            branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
            if(alter): branch_1 = BFP_out.bfp_out(branch_1, offset=ofs_layer[42], ShDepth=s_e_layer[41], MWidth=m_w_layer[41], EWidth=e_w_layer[41])
            branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[42], ShDepth=s_e_layer[42], MWidth=m_w_layer[42], EWidth=e_w_layer[42])
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            if(alter): branch_2 = BFP_out.bfp_out(branch_2, offset=ofs_layer[43], ShDepth=s_e_layer[43], MWidth=m_w_layer[43], EWidth=e_w_layer[43])
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            if(alter): branch_3 = BFP_out.bfp_out(branch_3, offset=ofs_layer[44], ShDepth=s_e_layer[44], MWidth=m_w_layer[44], EWidth=e_w_layer[44])
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[0], ShDepth=s_e_layer[3], MWidth=m_w_layer[3], EWidth=e_w_layer[3])
        end_points[end_point] = net
        if(debug): net = tf.Print(net, [net], "After seventh Inception Block: \n", summarize=999999999) 
        if final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool_5a_2x2'
        net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[45], ShDepth=s_e_layer[45], MWidth=m_w_layer[45], EWidth=e_w_layer[45])
            branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[46], ShDepth=s_e_layer[46], MWidth=m_w_layer[46], EWidth=e_w_layer[46])
            branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
            if(alter): branch_1 = BFP_out.bfp_out(branch_1, offset=ofs_layer[47], ShDepth=s_e_layer[47], MWidth=m_w_layer[47], EWidth=e_w_layer[47])
            branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[48], ShDepth=s_e_layer[48], MWidth=m_w_layer[48], EWidth=e_w_layer[48])
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            if(alter): branch_2 = BFP_out.bfp_out(branch_2, offset=ofs_layer[49], ShDepth=s_e_layer[49], MWidth=m_w_layer[49], EWidth=e_w_layer[49])
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0a_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            if(alter): branch_3 = BFP_out.bfp_out(branch_3, offset=ofs_layer[50], ShDepth=s_e_layer[50], MWidth=m_w_layer[50], EWidth=e_w_layer[50])
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[50], ShDepth=s_e_layer[3], MWidth=m_w_layer[3], EWidth=e_w_layer[3])
        end_points[end_point] = net
        if(debug): net = tf.Print(net, [net], "After eighth Inception Block: \n", summarize=999999999) 
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[51], ShDepth=s_e_layer[51], MWidth=m_w_layer[51], EWidth=e_w_layer[51])
            branch_0 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[52], ShDepth=s_e_layer[52], MWidth=m_w_layer[52], EWidth=e_w_layer[52])
            branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
            if(alter): branch_1 = BFP_out.bfp_out(branch_1, offset=ofs_layer[53], ShDepth=s_e_layer[53], MWidth=m_w_layer[53], EWidth=e_w_layer[53])
            branch_1 = slim.conv2d(branch_1, 384, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[54], ShDepth=s_e_layer[54], MWidth=m_w_layer[54], EWidth=e_w_layer[54])
            branch_2 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
            if(alter): branch_2 = BFP_out.bfp_out(branch_2, offset=ofs_layer[55], ShDepth=s_e_layer[55], MWidth=m_w_layer[55], EWidth=e_w_layer[55])
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            if(alter): branch_3 = BFP_out.bfp_out(branch_3, offset=ofs_layer[56], ShDepth=s_e_layer[56], MWidth=m_w_layer[56], EWidth=e_w_layer[56])
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        if(alter): net = BFP_out.bfp_out(net, offset=ofs_layer[57], ShDepth=s_e_layer[57], MWidth=m_w_layer[57], EWidth=e_w_layer[57])
        end_points[end_point] = net
        if(debug): net = tf.Print(net, [net], "After ninth Inception Block: \n", summarize=999999999) 
        if final_endpoint == end_point: return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v1(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV1',
                 global_pool=False,
                 m_w=23,
                 e_w=8,
                 s_w=3,
		 offset=0,
                 alter=False,
                 debug=False,
                 scale=False):
  """Defines the Inception V1 architecture.

  This architecture is defined in:

    Going deeper with convolutions
    Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    http://arxiv.org/pdf/1409.4842v1.pdf.

  The default image size used to train this network is 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is of
        shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    global_pool: Optional boolean flag to control the avgpooling before the
      logits layer. If false or unset, pooling is done with a fixed window
      that reduces default-sized inputs to 1x1, while larger inputs lead to
      larger outputs. If true, any input size is pooled down to 1x1.

  Returns:
    net: a Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped-out input to the logits layer
      if num_classes is 0 or None.
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  # Final pooling and prediction
  with tf.variable_scope(scope, 'InceptionV1', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_v1_base(inputs, scope=scope, m_w=m_w, s_w=s_w, e_w=e_w, alter=alter, debug=debug, scale=scale)
      with tf.variable_scope('Logits'):
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
          end_points['global_pool'] = net
        else:
          # Pooling with a fixed kernel size.
          net = slim.avg_pool2d(net, [7, 7], stride=1, scope='AvgPool_0a_7x7')
          end_points['AvgPool_0a_7x7'] = net
        if not num_classes:
          return net, end_points
        net = slim.dropout(net, dropout_keep_prob, scope='Dropout_0b')
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_0c_1x1')
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points
inception_v1.default_image_size = 224

inception_v1_arg_scope = inception_utils.inception_arg_scope
