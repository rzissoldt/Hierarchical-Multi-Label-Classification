# -*- coding:utf-8 -*-
__author__ = 'Ruben'
import tensorflow.compat.v1 as tf1
import tensorflow as tf
from official.vision.modeling.backbones.resnet import ResNet
tf1.disable_v2_behavior()
class ImageHARNN(object):
    """A HARNN for image classification."""

    def __init__(
            self,input_size,attention_unit_size,
            fc_hidden_size, num_classes_list, total_classes, l2_reg_lambda=0.0):

        # Placeholders for input, output, dropout_prob and training_tag
        self.resnet = ResNet(model_id=50,input_specs=tf.keras.layers.InputSpec(shape=[None,*input_size]))
        self.input_x = tf1.placeholder(tf1.float32, [None, *input_size], name="input_x")
        #self.input_ys = [tf1.placeholder(tf1.float32,[None,num_classes_list[i]],name="input_y_{0}".format(str(i))) for i in range(len(num_classes_list))]
        for i, num_classes in enumerate(num_classes_list):
            setattr(self, f"input_y_{i}", tf1.placeholder(tf1.float32, [None, num_classes], name=f"input_y_{i}"))
        self.input_y = tf1.placeholder(tf1.float32, [None, total_classes], name="input_y")
        self.dropout_keep_prob = tf1.placeholder(tf1.float32, name="dropout_keep_prob")
        self.alpha = tf1.placeholder(tf1.float32, name="alpha")
        self.is_training = tf1.placeholder(tf1.bool, name="is_training")

        self.global_step = tf1.Variable(0, trainable=False, name="Global_Step")

        def _attention(input_x, num_classes, name=""):
            """
            Attention Layer. Also known as TCA Module.

            Args:
                input_x: [batch_size, feature_length]
                num_classes: The number of i th level classes.
                name: Scope name.
            Returns:
                attention_weight: [batch_size, num_classes, feature_length]
                attention_out: [batch_size]
            """
            num_channels,spatial_dim = input_x.get_shape().as_list()[1:]
            with tf1.name_scope(name + "attention"):
                W_s1 = tf1.Variable(tf1.truncated_normal(shape=[attention_unit_size, spatial_dim],
                                                       stddev=0.1, dtype=tf1.float32), name="W_s1")
                W_s2 = tf1.Variable(tf1.truncated_normal(shape=[num_classes, attention_unit_size],
                                                       stddev=0.1, dtype=tf1.float32), name="W_s2")
                # attention_matrix: [batch_size, num_classes, sequence_length]
                attention_matrix = tf1.map_fn(
                    fn=lambda x: tf1.matmul(W_s2, x),
                    elems=tf1.tanh(
                        tf1.map_fn(
                            fn=lambda x: tf1.matmul(W_s1, tf.transpose(x)),
                            elems=input_x,
                            dtype=tf1.float32
                        )
                    )
                )
                attention_weight = tf1.nn.softmax(attention_matrix, name="attention")
                attention_out = tf1.matmul(attention_weight, input_x)
                attention_out = tf1.reduce_mean(attention_out, axis=1)
            return attention_weight, attention_out

        def _fc_layer(input_x, name=""):
            """
            Fully Connected Layer. Getting used to calculate the local Predictions, used for Class Prediction Module

            Args:
                input_x: [batch_size, *]
                name: Scope name.
            Returns:
                fc_out: [batch_size, fc_hidden_size]
            """
            with tf1.name_scope(name + "fc"):
                num_units = input_x.get_shape().as_list()[-1]
                W = tf1.Variable(tf1.truncated_normal(shape=[num_units, fc_hidden_size],
                                                    stddev=0.1, dtype=tf1.float32), name="W")
                b = tf1.Variable(tf1.constant(value=0.1, shape=[fc_hidden_size], dtype=tf1.float32), name="b")
                fc = tf1.nn.xw_plus_b(input_x, W, b)
                fc_out = tf1.nn.relu(fc)
            return fc_out

        def _local_layer(input_x, input_att_weight, num_classes, name=""):
            """
            Local Layer. Used for Class Depency Module

            Args:
                input_x: [batch_size, fc_hidden_size]
                input_att_weight: [batch_size, num_classes, feature_length]
                num_classes: Number of classes.
                name: Scope name.
            Returns:
                logits: [batch_size, num_classes]
                scores: [batch_size, num_classes]
                visual: [batch_size, feature_length]
            """
            with tf1.name_scope(name + "output"):
                num_units = input_x.get_shape().as_list()[-1]
                W = tf1.Variable(tf1.truncated_normal(shape=[num_units, num_classes],
                                                    stddev=0.1, dtype=tf1.float32), name="W")
                b = tf1.Variable(tf1.constant(value=0.1, shape=[num_classes], dtype=tf1.float32), name="b")
                logits = tf1.nn.xw_plus_b(input_x, W, b, name="logits")
                scores = tf1.sigmoid(logits, name="scores")

                # shape of visual: [batch_size, sequence_length]
                visual = tf1.multiply(input_att_weight, tf1.expand_dims(scores, -1))
                visual = tf1.nn.softmax(visual)
                visual = tf1.reduce_mean(visual, axis=1, name="visual")
            return logits, scores, visual

        def _linear(input_, output_size, initializer=None, scope="SimpleLinear"):
            """
            Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k].

            Args:
                input_: a tensor or a list of 2D, batch x n, Tensors.
                output_size: int, second dimension of W[i].
                initializer: The initializer.
                scope: VariableScope for the created subgraph; defaults to "SimpleLinear".
            Returns:
                A 2D Tensor with shape [batch x output_size] equal to
                sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
            Raises:
                ValueError: if some of the arguments has unspecified or wrong shape.
            """

            shape = input_.get_shape().as_list()
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: {0}".format(str(shape)))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: {0}".format(str(shape)))
            input_size = shape[1]

            # Now the computation.
            with tf1.variable_scope(scope):
                W = tf1.get_variable("W", [input_size, output_size], dtype=input_.dtype)
                b = tf1.get_variable("b", [output_size], dtype=input_.dtype, initializer=initializer)

            return tf1.nn.xw_plus_b(input_, W, b)

        def _highway_layer(input_, size, num_layers=1, bias=-2.0):
            """
            Highway Network (cf. http://arxiv.org/abs/1505.00387).
            t = sigmoid(Wx + b); h = relu(W'x + b')
            z = t * h + (1 - t) * x
            where t is transform gate, and (1 - t) is carry gate.
            """

            for idx in range(num_layers):
                h = tf1.nn.relu(_linear(input_, size, scope=("highway_h_{0}".format(idx))))
                t = tf1.sigmoid(_linear(input_, size, initializer=tf1.constant_initializer(bias),
                                       scope=("highway_t_{0}".format(idx))))
                output = t * h + (1. - t) * input_
                input_ = output

            return output
        """
        TODO: IMAGE FEATURE EXTRACTOR 
        """
        
        self.feature_extractor_out = self.resnet(self.input_x)['5']
        spatial_dim1, spatial_dim2, num_channels = self.feature_extractor_out.get_shape().as_list()[1:]
        self.feature_extractor_out = tf.transpose(tf.reshape(self.feature_extractor_out,[-1,spatial_dim1*spatial_dim2,num_channels]),perm=[0,2,1])
        self.feature_extractor_out_pool= tf1.reduce_mean(self.feature_extractor_out,axis=1)
        
        self.att_weights = []
        self.att_outs = []
        self.local_inputs = []
        self.local_fc_outs = []
        self.logits_list = []
        self.scores_list = []
        self.visual_list = []
        """
        This part is used to dynamically generate the HAM Modules. 
        """
        for i in range(len(num_classes_list)):
            if i == 0:
                # First Level
                att_weight, att_out = _attention(self.feature_extractor_out, num_classes_list[i], name="{0}-".format(str(i)))
                local_input = tf1.concat([self.feature_extractor_out_pool, att_out], axis=1)
                local_fc_out = _fc_layer(local_input, name="{0}-local-".format(str(i)))
                logits, scores, visual = _local_layer(local_fc_out, att_weight, num_classes_list[0], name="{0}-".format(str(i)))
                self.logits_list.append(logits)
                self.scores_list.append(scores)
                self.local_fc_outs.append(local_fc_out)
                self.visual_list.append(visual)
            else:
                # Second Level
                att_input = tf1.multiply(self.feature_extractor_out, tf1.expand_dims(self.visual_list[-1], -1))
                att_weight, att_out = _attention(att_input, num_classes_list[i], name="{0}-".format(str(i)))
                local_input = tf1.concat([self.feature_extractor_out_pool, att_out], axis=1)
                local_fc_out = _fc_layer(local_input, name="{0}-local-".format(str(i)))
                logits, scores, visual = _local_layer(local_fc_out, att_weight, num_classes_list[i], name="{0}-".format(str(i)))
                self.logits_list.append(logits)
                self.scores_list.append(scores)
                self.local_fc_outs.append(local_fc_out)
                self.visual_list.append(visual)

        # Concat
        # shape of ham_out: [batch_size, fc_hidden_size * 4]
        self.ham_out = tf1.concat(self.local_fc_outs, axis=1)

        # Fully Connected Layer
        self.fc_out = _fc_layer(self.ham_out)

        # Highway Layer
        with tf1.name_scope("highway"):
            self.highway = _highway_layer(self.fc_out, self.fc_out.get_shape()[1], num_layers=1, bias=0)

        # Add dropout
        with tf1.name_scope("dropout"):
            self.h_drop = tf1.nn.dropout(self.highway, self.dropout_keep_prob)

        # Global scores
        with tf1.name_scope("global-output"):
            num_units = self.h_drop.get_shape().as_list()[-1]
            W = tf1.Variable(tf1.truncated_normal(shape=[num_units, total_classes],
                                                stddev=0.1, dtype=tf1.float32), name="W")
            b = tf1.Variable(tf1.constant(value=0.1, shape=[total_classes], dtype=tf1.float32), name="b")
            self.global_logits = tf1.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.global_scores = tf1.sigmoid(self.global_logits, name="scores")

        with tf1.name_scope("output"):
            self.local_scores = tf1.concat(self.scores_list, axis=1)
            self.scores = tf1.add(self.alpha * self.global_scores, (1 - self.alpha) * self.local_scores, name="scores")

        # Calculate mean cross-entropy loss, L2 loss
        with tf1.name_scope("loss"):
            def cal_loss(labels, logits, name):
                losses = tf1.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
                losses = tf1.reduce_mean(tf1.reduce_sum(losses, axis=1), name=name + "losses")
                return losses

            # Local Loss
            losses = []
            for i in range(len(num_classes_list)):
                input_y_attr = getattr(self, f"input_y_{i}")  # Dynamically get the attribute
                logits_attr = self.logits_list[i]  # Assuming logits_list is a list of logits corresponding to each input_y
                loss = cal_loss(labels=input_y_attr, logits=logits_attr, name=f"{i}_")  # Perform loss calculation
                losses.append(loss)
            local_losses = tf1.add_n(losses, name="local_losses")

            # Global Loss
            global_losses = cal_loss(labels=self.input_y, logits=self.global_logits, name="global_")

            # L2 Loss
            l2_losses = tf1.add_n([tf1.nn.l2_loss(tf1.cast(v, tf1.float32)) for v in tf1.trainable_variables()],
                                 name="l2_losses") * l2_reg_lambda
            self.loss = tf1.add_n([local_losses, global_losses, l2_losses], name="loss")
            
    
