import tensorflow as tf

class TIA_Contrastive(tf.keras.Model):
    def __init__(self, c_in, nmb_prototype, batch_size, tau=0.5):
        super(TIA_Contrastive, self).__init__()
        self.l2norm = lambda x: tf.math.l2_normalize(x, axis=1)
        self.prototypes = tf.keras.layers.Dense(units=nmb_prototype, input_shape=(c_in,), use_bias=False)
        self.nmb_prototype = nmb_prototype
        self.batch_size = batch_size
        self.tau = tau
        self.c_in = c_in
        self.weights_init()

    def weights_init(self):
        self.prototypes.build((None, self.c_in))
        initializer = tf.keras.initializers.glorot_uniform()
        self.prototypes.kernel.assign(initializer(self.prototypes.kernel.shape))

    def sinkhorn(self, out, epsilon=0.05, sinkhorn_iterations=3):
        Q = tf.exp(out / epsilon)
        B = self.batch_size
        K = tf.cast(out.shape[1], tf.float32)

        sum_Q = tf.reduce_sum(Q)
        Q /= sum_Q
        for it in range(sinkhorn_iterations):
            Q /= tf.reduce_sum(Q, axis=1, keepdims=True)
            Q /= K
            Q /= tf.reduce_sum(Q, axis=0, keepdims=True)
            Q /= B

        Q *= B
        return tf.transpose(Q)

    def clone_weights(self):
        weights = [tf.identity(w) for w in self.prototypes.get_weights()]
        return weights

    def call(self, z1, z2):
        z1 = tf.reshape(z1, (-1, self.c_in))
        z1 = self.l2norm(z1)
        zc1 = self.prototypes(z1)

        zc2 = self.prototypes(self.l2norm(tf.reshape(z2, (-1, self.c_in))))

        q1 = self.sinkhorn(tf.stop_gradient(zc1))
        q2 = self.sinkhorn(tf.stop_gradient(zc2))
        q1 = tf.reshape(q1, shape=[-1, q1.shape[0]])
        q2 = tf.reshape(q2, shape=[-1, q2.shape[0]])

        l1_z = tf.nn.log_softmax(zc2 / self.tau, axis=1)
        l1_z = tf.reduce_sum(q1 * l1_z, axis=1)
        l1 = -tf.reduce_mean(l1_z)
        l2 = -tf.reduce_mean(tf.reduce_sum(q2 * tf.nn.log_softmax(zc1 / self.tau, axis=1), axis=1))

        return l1 + l2
