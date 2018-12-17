import tensorflow as tf
import numpy as np
import sklearn as sk

class TextCNN(object):
    #用于文本分类的CNN, 使用嵌入层，然后是卷积，最大池和softmax层。
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # 输入，输出和丢失的占位符
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        #跟踪l2正则化损失（可选）
        l2_loss = tf.constant(0.0)

        #嵌入图层
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # 为每个过滤器大小创建一个卷积+ 最大池化层
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                #卷积层
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                #应用非线性
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="sigmod")
                #最大化输出
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        #结合所有池化功能
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # 添加dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # 最终（非标准化）分数和预测
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
            shape = [num_filters_total, num_classes],
            initializer = tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

            #计算平均交叉熵损失
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        with tf.name_scope("accuracy"):
            y_pred = self.predictions
            y = self.input_y
            correct_predictions = tf.equal(y_pred, tf.argmax(y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            predicted = y_pred
            actual = tf.argmax(y, 1)
            TP = tf.count_nonzero(predicted * actual)
            # TN = tf.count_nonzero((predicted - 1) * (actual - 1))
            FP = tf.count_nonzero(predicted * (actual - 1))
            FN = tf.count_nonzero((predicted - 1) * actual)

            self.precision = TP / (TP + FP)
            self.recall = TP / (TP + FN)
            self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)
        #准确性


# self.matric = confusion_matrix(self.predictions, tf.argmax(self.input_y, 1))
# self.recall = recall_score(self.predictions, tf.argmax(self.input_y, 1))
# self.precision = precision_score(self.predictions, tf.argmax(self.input_y, 1))