# -*- coding:utf-8 -*-
__author__ = 'Ruben'

import os
import sys
import time
import logging
import solt as slt
sys.path.append('../')
logging.getLogger('tensorflow').disabled = True
from utils import xtree_utils as xtree
import tensorflow.compat.v1 as tf1
import numpy as np
import tensorflow as tf
from image_harnn import ImageHARNN
from utils import checkmate as cm
from utils import data_helpers as dh
from utils import param_parser as parser
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from solt.transforms import (
    Flip,
    Rotate,
    Scale,
    Shear,
    CvtColor,
    HSV,
    Blur,
    CutOut,
    Resize
)
from solt.core import Stream, SelectiveStream

# Define the augmentation pipeline
args = parser.image_parameter_parser()
# Checks if GPU Support ist active
if not args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
OPTION = dh._option(pattern=0)
logger = dh.logger_fn("tflog", "logs/{0}-{1}.log".format('Train','test'))## if OPTION == 'T' else 'Restore', time.asctime()))



def create_input_data(data: dict):
    # Extract dynamic keys with the prefix 'layer'
    layer_keys = [key for key in data.keys() if key.startswith('layer')]

    # Extract corresponding lists from the dictionary based on layer keys
    layers_data = [data[key] for key in layer_keys]

    # Zip all lists together
    return zip(data['file_names'],data['onehot_labels'], *layers_data)


def train_image_harnn():
    """Training Image HARNN model."""
    # Print parameters used for the model
    dh.tab_printer(args, logger)

    # Load sentences, labels, and training parameters
    logger.info("Loading data...")
    logger.info("Data processing...")
    
    hierarchy = xtree.load_xtree_json(args.hierarchy_file)
    hierarchy_dicts = xtree.generate_dicts_per_level(hierarchy)
    train_data = dh.load_image_data_and_labels(args.train_file,hierarchy_dicts)
    val_data = dh.load_image_data_and_labels(args.validation_file, hierarchy_dicts)

    image_dir = args.image_dir
    input_size = args.input_size
    def _get_num_classes_from_hierarchy(hierarchy_dicts):
        return [len(hierarchy_dict.keys()) for hierarchy_dict in hierarchy_dicts]
    
    num_classes_list = _get_num_classes_from_hierarchy(hierarchy_dicts)
    total_classes = sum(num_classes_list) 
    # Build a graph and image_harnn object
    with tf.Graph().as_default():
        session_conf = tf1.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
        session_conf.gpu_options.allow_growth = args.gpu_options_allow_growth
        sess = tf1.Session(config=session_conf)
        with sess.as_default():
            image_harnn = ImageHARNN(
                attention_unit_size=args.attention_dim,
                fc_hidden_size=args.fc_dim,
                num_classes_list=num_classes_list,
                total_classes=total_classes,
                l2_reg_lambda=args.l2_lambda,
                input_size=input_size
            )

            # Define training procedure
            with tf1.control_dependencies(tf1.get_collection(tf1.GraphKeys.UPDATE_OPS)):
                learning_rate = tf1.train.exponential_decay(learning_rate=args.learning_rate,
                                                           global_step=image_harnn.global_step,
                                                           decay_steps=args.decay_steps,
                                                           decay_rate=args.decay_rate,
                                                           staircase=True)
                optimizer = tf1.train.AdamOptimizer(learning_rate)
                grads, vars = zip(*optimizer.compute_gradients(image_harnn.loss))
                grads, _ = tf1.clip_by_global_norm(grads, clip_norm=args.norm_ratio)
                train_op = optimizer.apply_gradients(zip(grads, vars), global_step=image_harnn.global_step, name="train_op")

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in zip(grads, vars):
                if g is not None:
                    grad_hist_summary = tf1.summary.histogram("{0}/grad/hist".format(v.name), g)
                    sparsity_summary = tf1.summary.scalar("{0}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf1.summary.merge(grad_summaries)

            # Output directory for models and summaries
            out_dir = dh.get_out_dir(OPTION, logger)
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            best_checkpoint_dir = os.path.abspath(os.path.join(out_dir, "bestcheckpoints"))

            # Summaries for loss
            loss_summary = tf1.summary.scalar("loss", image_harnn.loss)

            # Train summaries
            train_summary_op = tf1.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf1.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf1.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf1.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf1.train.Saver(tf1.global_variables(), max_to_keep=args.num_checkpoints)
            best_saver = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir, num_to_keep=3, maximize=True)

            #Augmentation Stream
            augmentation_train_pipeline = Stream([
                Rotate(angle_range=(-90, 90), p=1, padding='r'),
                Flip(axis=1, p=0.5),
                Flip(axis=0, p=0.5),
                Shear(range_x=0.3, range_y=0.8, p=0.5, padding='r'),
                Scale(range_x=(0.8, 1.3), padding='r', range_y=(0.8, 1.3), same=False, p=0.5),
                HSV((0, 10), (0, 10), (0, 10)),
                Blur(k_size=7, blur_type='m'),
                SelectiveStream([
                    CutOut(40, p=1),
                CutOut(50, p=1),
                CutOut(10, p=1),
                Stream(),
                Stream(),
                ], n=3),
                Resize(input_size[:2])
            ], ignore_fast_mode=True)
            if OPTION == 'R':
                # Load image_harnn model
                logger.info("Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                logger.info(checkpoint_file)

                # Load the saved meta graph and restore variables
                saver = tf1.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            if OPTION == 'T':
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf1.global_variables_initializer())
                sess.run(tf1.local_variables_initializer())

                # Save the embedding visualization
                #saver.save(sess, os.path.join(out_dir, "embedding", "embedding.ckpt"))

            current_step = sess.run(image_harnn.global_step)

            def train_step(batch_data,input_size,image_dir):
                """A single training step."""
                file_names, y_onehots, *unzipped_data = zip(*batch_data)
                file_paths = [os.path.join(image_dir,file_name) for file_name in file_names]
                yss = unzipped_data
                images = tuple(dh.load_preprocess_augment_images(file_paths,input_size,augmentation_train_pipeline))
                feed_dict = {
                    image_harnn.input_x: images,
                    image_harnn.input_y: y_onehots,
                    image_harnn.dropout_keep_prob: args.dropout_rate,
                    image_harnn.freeze_backbone: args.freeze_backbone,
                    image_harnn.alpha: args.alpha,
                    image_harnn.is_training: True
                }
                for i in range(len(yss)):
                    key = 'input_y_{0}'.format(i)
                    feed_dict[getattr(image_harnn, key)] = yss[i]

                _, step, summaries, loss = sess.run(
                    [train_op, image_harnn.global_step, train_summary_op, image_harnn.loss], feed_dict)
                logger.info("step {0}: loss {1:g}".format(step, loss))
                train_summary_writer.add_summary(summaries, step)

            def validation_step(val_loader,input_size,image_dir, writer=None):
                """Evaluates model on a validation set."""
                batches_validation = dh.batch_iter(list(create_input_data(val_loader)), args.batch_size, 1)

                # Predict classes by threshold or topk ('ts': threshold; 'tk': topk)
                eval_counter, eval_loss = 0, 0.0
                eval_pre_tk = [0.0] * args.topK
                eval_rec_tk = [0.0] * args.topK
                eval_F1_tk = [0.0] * args.topK

                true_onehot_labels = []
                predicted_onehot_scores = []
                predicted_onehot_labels_ts = []
                predicted_onehot_labels_tk = [[] for _ in range(args.topK)]

                for batch_validation in batches_validation:
                    file_names, y_onehots, *unzipped_data = zip(*batch_validation)
                    file_paths = [os.path.join(image_dir,file_name) for file_name in file_names]
                    yss = unzipped_data
                    images = dh.load_preprocess_images(image_paths=file_paths,input_size=input_size)
                    feed_dict = {
                        image_harnn.input_x: images,
                        image_harnn.input_y: y_onehots,
                        image_harnn.dropout_keep_prob: args.dropout_rate,
                        image_harnn.freeze_backbone: args.freeze_backbone,
                        image_harnn.alpha: args.alpha,
                        image_harnn.is_training: False
                    }
                    for i in range(len(yss)):
                        key = 'input_y_{0}'.format(i)
                        feed_dict[getattr(image_harnn, key)] = yss[i]
                    step, summaries, scores, cur_loss = sess.run(
                        [image_harnn.global_step, validation_summary_op, image_harnn.scores, image_harnn.loss], feed_dict)
                    # Prepare for calculating metrics
                    for i in y_onehots:
                        true_onehot_labels.append(i)
                    for j in scores:
                        predicted_onehot_scores.append(j)
                    # Predict by threshold
                    batch_predicted_onehot_labels_ts = \
                        dh.get_onehot_label_threshold(scores=scores, threshold=args.threshold)
                    for k in batch_predicted_onehot_labels_ts:
                        predicted_onehot_labels_ts.append(k)
                    # Predict by topK
                    for top_num in range(args.topK):
                        batch_predicted_onehot_labels_tk = dh.get_onehot_label_topk(scores=scores, top_num=top_num+1)
                        for i in batch_predicted_onehot_labels_tk:
                            predicted_onehot_labels_tk[top_num].append(i)
                    eval_loss = eval_loss + cur_loss
                    eval_counter = eval_counter + 1
                    
                    if writer:
                        writer.add_summary(summaries, step)

                eval_loss = float(eval_loss / eval_counter)
                # Calculate Precision & Recall & F1
                eval_pre_ts = precision_score(y_true=np.array(true_onehot_labels),
                                              y_pred=np.array(predicted_onehot_labels_ts), average='micro')
                eval_rec_ts = recall_score(y_true=np.array(true_onehot_labels),
                                           y_pred=np.array(predicted_onehot_labels_ts), average='micro')
                eval_F1_ts = f1_score(y_true=np.array(true_onehot_labels),
                                      y_pred=np.array(predicted_onehot_labels_ts), average='micro')
                for top_num in range(args.topK):
                    eval_pre_tk[top_num] = precision_score(y_true=np.array(true_onehot_labels),
                                                           y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                           average='micro')
                    eval_rec_tk[top_num] = recall_score(y_true=np.array(true_onehot_labels),
                                                        y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                        average='micro')
                    eval_F1_tk[top_num] = f1_score(y_true=np.array(true_onehot_labels),
                                                   y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                   average='micro')
                # Calculate the average AUC
                eval_auc = roc_auc_score(y_true=np.array(true_onehot_labels),
                                         y_score=np.array(predicted_onehot_scores), average='micro')
                # Calculate the average PR
                eval_prc = average_precision_score(y_true=np.array(true_onehot_labels),
                                                   y_score=np.array(predicted_onehot_scores), average='micro')

                return eval_loss, eval_auc, eval_prc, eval_pre_ts, eval_rec_ts, eval_F1_ts, \
                       eval_pre_tk, eval_rec_tk, eval_F1_tk

            # Generate batches
            batches_train = dh.batch_iter(data=list(create_input_data(train_data)), batch_size=args.batch_size, num_epochs=args.epochs)
            
            num_batches_per_epoch = int((len(train_data['file_names']) - 1) / args.batch_size) + 1
            print(num_batches_per_epoch)
            # Training loop. For each batch...
            for batch_train in batches_train:
                train_step(batch_train,input_size,image_dir)
                current_step = tf1.train.global_step(sess, image_harnn.global_step)

                if current_step % args.evaluate_steps == 0:
                    logger.info("\nEvaluation:")
                    eval_loss, eval_auc, eval_prc, \
                    eval_pre_ts, eval_rec_ts, eval_F1_ts, eval_pre_tk, eval_rec_tk, eval_F1_tk = \
                        validation_step(val_data,input_size,image_dir, writer=validation_summary_writer)
                    logger.info("All Validation set: Loss {0:g} | AUC {1:g} | AUPRC {2:g}"
                                .format(eval_loss, eval_auc, eval_prc))
                    # Predict by threshold
                    logger.info("Predict by threshold: Precision {0:g}, Recall {1:g}, F1 {2:g}"
                                .format(eval_pre_ts, eval_rec_ts, eval_F1_ts))
                    # Predict by topK
                    logger.info("Predict by topK:")
                    for top_num in range(args.topK):
                        logger.info("Top{0}: Precision {1:g}, Recall {2:g}, F1 {3:g}"
                                    .format(top_num+1, eval_pre_tk[top_num], eval_rec_tk[top_num], eval_F1_tk[top_num]))
                    best_saver.handle(eval_prc, sess, current_step)
                if current_step % args.checkpoint_steps == 0:
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("Saved model checkpoint to {0}\n".format(path))
                if current_step % num_batches_per_epoch == 0:
                    current_epoch = current_step // num_batches_per_epoch
                    logger.info("Epoch {0} has finished!".format(current_epoch))

    logger.info("All Done.")


if __name__ == '__main__':
    train_image_harnn()