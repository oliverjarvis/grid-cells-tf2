# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Supervised training for the Grid cell network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfb
import sys
import time
print(sys.path)
print(sys.executable)
#import Tkinter  # pylint: disable=unused-import

#matplotlib.use('Agg')

import dataset_reader  # pylint: disable=g-bad-import-order, g-import-not-at-top
import model  # pylint: disable=g-bad-import-order
from model import GridCellNetwork
import scores  # pylint: disable=g-bad-import-order
import utils  # pylint: disable=g-bad-import-order


# Task config
FLAGS = {
    'task_dataset_info':'square_room',
    'task_root':'../data',
    'task_env_size':2.2,
    'task_n_pc':[256],
    'task_pc_scale':[0.01],
    'task_n_hdc':[12],
    'task_hdc_concentration':[20.],
    'task_neurons_seed':8341,
    'task_targets_type':'softmax',
    'task_lstm_init_type':'softmax',
    'task_velocity_inputs':True,
    'task_velocity_noise':[0.0,0.0,0.0],
    'model_nh_lstm':128,
    'model_nh_bottleneck':256,
    'model_dropout_rates':[0.5],
    'model_weight_decay':1e-5,
    'model_bottleneck_has_bias':False,
    'model_init_weight_disp':0.0,
    'training_epochs':1000,
    'training_steps_per_epoch':1000,
    'training_minibatch_size':10,
    'training_evaluation_minibatch_size':4000,
    'training_clipping_function':'utils.clip_all_gradients',
    'training_clipping':1e-5,
    'training_optimizer_class':'tf.train.RMSPropOptimizer',
    'training_optimizer_options':{"learning_rate": 1e-5,"momentum": 0.9},
    'saver_results_directory':"results",
    'saver_eval_time':2
}

preprocess_time = 0


def train():
    """Training loop."""

    #tf.reset_default_graph()

    # Create the motion models for training and evaluation
    data_reader = dataset_reader.DataReader(
        FLAGS['task_dataset_info'], root=FLAGS['task_root'], batch_size=FLAGS['training_minibatch_size'])
    dataset = data_reader.read()
    #train_traj = data_reader.read()
    # Create the ensembles that provide targets during training
    place_cell_ensembles = utils.get_place_cell_ensembles(
        env_size=FLAGS['task_env_size'],
        neurons_seed=FLAGS['task_neurons_seed'],
        targets_type=FLAGS['task_targets_type'],
        lstm_init_type=FLAGS['task_lstm_init_type'],
        n_pc=FLAGS['task_n_pc'],
        pc_scale=FLAGS['task_pc_scale'])

    head_direction_ensembles = utils.get_head_direction_ensembles(
        neurons_seed=FLAGS['task_neurons_seed'],
        targets_type=FLAGS['task_targets_type'],
        lstm_init_type=FLAGS['task_lstm_init_type'],
        n_hdc=FLAGS['task_n_hdc'],
        hdc_concentration=FLAGS['task_hdc_concentration'])

    target_ensembles = place_cell_ensembles + head_direction_ensembles

    '''
    # Get a trajectory batch
    input_tensors = []
    init_pos, init_hd, ego_vel, target_pos, target_hd = train_traj
    if FLAGS['task_velocity_inputs']:
        # Add the required amount of noise to the velocities
        vel_noise = tfb.distributions.Normal(0.0, 1.0).sample(
            sample_shape=ego_vel.get_shape()) * FLAGS['task_velocity_noise']
        input_tensors = [ego_vel + vel_noise] + input_tensors
    # Concatenate all inputs
    inputs = tf.concat(input_tensors, axis=2)

    # Replace euclidean positions and angles by encoding of place and hd ensembles
    # Note that the initial_conds will be zeros if the ensembles were configured
    # to provide that type of initialization
    initial_conds = utils.encode_initial_conditions(
        init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)

    # Encode targets as well
    ensembles_targets = utils.encode_targets(
        target_pos, target_hd, place_cell_ensembles, head_direction_ensembles)

    # Estimate future encoding of place and hd ensembles inputing egocentric vels
    '''

    #Defining model
    grid_cell_model = GridCellNetwork(
        target_ensembles = target_ensembles,
        nh_lstm = FLAGS['model_nh_lstm'],
        nh_bottleneck = FLAGS['model_nh_bottleneck'],
        dropoutrates_bottleneck = FLAGS['model_dropout_rates'],
        bottleneck_weight_decay = FLAGS['model_weight_decay'],
        bottleneck_has_bias = FLAGS['model_bottleneck_has_bias'],
        init_weight_disp = FLAGS['model_init_weight_disp'],
    )


    # Store the grid scores
    grid_scores = dict()
    grid_scores['btln_60'] = np.zeros((FLAGS['model_nh_bottleneck'],))
    grid_scores['btln_90'] = np.zeros((FLAGS['model_nh_bottleneck'],))
    grid_scores['btln_60_separation'] = np.zeros((FLAGS['model_nh_bottleneck'],))
    grid_scores['btln_90_separation'] = np.zeros((FLAGS['model_nh_bottleneck'],))
    grid_scores['lstm_60'] = np.zeros((FLAGS['model_nh_lstm'],))
    grid_scores['lstm_90'] = np.zeros((FLAGS['model_nh_lstm'],))

    # Create scorer objects
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    masks_parameters = zip(starts, ends.tolist())
    latest_epoch_scorer = scores.GridScorer(20, data_reader.get_coord_range(),
                                            masks_parameters)

# we can run without feed_dict either because we are in the singular monitored session
# so we use fetches on sess.run
# pos_xy is simply the input data. In tf1 it's the computed graph up until target_pos, which is very early and basically just fetched the data.

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5, momentum=0.9, clipvalue=FLAGS['training_clipping'])
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


    pos_xy = []
    bottleneck = []
    lstm_output = []

    @tf.function
    def loss_function(targets, logits):
        pc_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=targets[0], logits=logits[0], name='pc_loss')
        hd_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=targets[1], logits=logits[1], name='hd_loss')
        total_loss = pc_loss + hd_loss
        return tf.reduce_mean(total_loss, name='train_loss')
    @tf.function
    def train_step(velocities, targets, logits):   
        global preprocess_time     
        with tf.GradientTape() as tape:
            predictions,_ = grid_cell_model(velocities, initial_conds, trainable=True)
            ensembles_logits, bottleneck, lstm_output = predictions
            loss = loss_function(targets, ensembles_logits)
        gradients = tape.gradient(loss, grid_cell_model.trainable_weights)

        optimizer.apply_gradients(zip(gradients, grid_cell_model.trainable_weights))
        
        train_loss(loss)
        return {"bottleneck":bottleneck, "lstm_output":lstm_output, "pos_xy":target_pos}
    for epoch in range(FLAGS['training_epochs']):
        loss_acc = list()
        res = dict()
        #for _ in range(FLAGS['training_steps_per_epoch']):
        train_loss.reset_states()
        for batch, train_trajectory in enumerate(dataset):
            print(batch)
            '''some preprocessing that maybe should be done in the data_pipeline'''
            start_time = time.time()
            init_pos = train_trajectory['init_pos']
            init_hd = train_trajectory['init_hd']
            ego_vel = train_trajectory['ego_vel']
            target_pos = train_trajectory['target_pos']
            target_hd = train_trajectory['target_hd']
            input_tensors = []
            if FLAGS['task_velocity_inputs']:
                # Add the required amount of noise to the velocities
                vel_noise = tfb.distributions.Normal(0.0, 1.0).sample(
                    sample_shape=tf.shape(ego_vel)) * FLAGS['task_velocity_noise']
                input_tensors = [ego_vel + vel_noise] + input_tensors
            velocities = tf.concat(input_tensors, axis=2)
            initial_conds = utils.encode_initial_conditions(init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)
            ensembles_targets = utils.encode_targets(target_pos, target_hd, place_cell_ensembles, head_direction_ensembles)
            mb_res = train_step(velocities, ensembles_targets, initial_conds)
            #res = utils.concat_dict(res, mb_res)

            if batch % 1000 > 600:
                pos_xy.append(mb_res['pos_xy'])
                bottleneck.append(mb_res['bottleneck'])
                lstm_output.append(mb_res['lstm_output'])

            if batch % 1000 == 0 and batch != 0:
                print(preprocess_time)
                print('Epoch {}, batch {}, loss {}'.format(epoch, batch, train_loss.result()))
                for i in range(len(pos_xy)):
                    mb_res = {"bottleneck":bottleneck[i], "lstm_out":lstm_output[i], "pos_xy":pos_xy[i]}
                    utils.concat_dict(res, mb_res)
                pos_xy = []
                bottleneck = []
                lstm_output = []
                mb_res = dict()
                # Store at the end of validation
                filename = 'rates_and_sac_latest_hd.pdf'
                grid_scores['btln_60'], grid_scores['btln_90'], grid_scores[
                    'btln_60_separation'], grid_scores[
                        'btln_90_separation'] = utils.get_scores_and_plot(
                            latest_epoch_scorer, res['pos_xy'], res['bottleneck'],
                            FLAGS['saver_results_directory'], filename)
                res = dict()
            

def main():
  #tf.logging.set_verbosity(3)  # Print INFO log messages.
  train()

if __name__ == '__main__':
  main()