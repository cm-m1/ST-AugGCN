import tensorflow as tf
import numpy as np
import math

class Align(tf.keras.layers.Layer):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = tf.keras.layers.Conv2D(c_out, (1, 1))  # filter=(1,1), similar to fc

    def call(self, x):  # x: (n,c,l,v)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, self.c_out - self.c_in]])
            return tf.pad(x, paddings)
        return x

class SpatioConvLayer(tf.keras.layers.Layer):
    def __init__(self, ks, c_in, c_out):
        super(SpatioConvLayer, self).__init__()
        self.theta = tf.Variable(
            tf.random.uniform((c_in, c_out), -math.sqrt(5), math.sqrt(5)))  # kernel: C_in*C_out*ks
        self.b = tf.Variable(tf.random.uniform((1, c_out), 1, 1))
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        initializer_theta = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        self.theta.assign(initializer_theta(self.theta.shape))
        initializer_b = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='uniform')
        fan_in_b = self.b.shape.as_list()[1]
        bound = 1 / math.sqrt(fan_in_b)
        self.b.assign(tf.random.uniform(self.b.shape, minval=-bound, maxval=bound))

    def call(self, inputs, extra_param=None):
        x_c = tf.einsum("knm,bn->bm", extra_param, inputs)
        x_gc = tf.einsum("io,bm->bo", self.theta, x_c) + self.b
        return tf.nn.relu(x_gc)


def cal_laplacian(graph):
    graph = graph.A
    I = np.eye(graph.shape[0], dtype=graph.dtype)
    graph = graph + I
    D = np.diag(graph.sum(axis=1) ** (-0.5))
    L = I - np.matmul(np.matmul(D, graph), D)
    return L


def cheb_polynomial(laplacian, K):
    """
    :param laplacian: the graph laplacian, [v, v].
    :return: the multi order Chebyshev laplacian, [K, v, v].
    """
    N = laplacian.shape[0]
    multi_order_laplacian = np.zeros([K, N, N], dtype=np.float)
    multi_order_laplacian[0] = np.eye(N, dtype=np.float)

    if K == 1:
        return multi_order_laplacian
    else:
        multi_order_laplacian[1] = laplacian
        if K == 2:
            return multi_order_laplacian
        else:
            for k in range(2, K):
                multi_order_laplacian[k] = 2 * np.matmul(laplacian, multi_order_laplacian[k - 1]) - \
                                           multi_order_laplacian[k - 2]

    return multi_order_laplacian