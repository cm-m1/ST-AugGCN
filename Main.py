import tensorflow as tf
import pandas as pd
import math
import os
import time
import numpy.linalg as la
from tqdm import tqdm
from Quary_GraphAdj import *
from Multimodal_input import preprocess_data
from TrafficTIACell import TIACell
from EventDataset.WeatherEvents.AccompanyEncoding import get_weather_map
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

time_start = time.time()

# Define flags for TensorFlow
flags = tf.app.flags
FLAGS = flags.FLAGS

# Define flags with default values
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('training_epoch', 500, 'Number of epochs to train.')
flags.DEFINE_integer('units', 64, 'Hidden units of GRU.')
flags.DEFINE_integer('seq_len', 4, 'Time length of inputs.')
flags.DEFINE_integer('pre_len', 1, 'Time length of prediction.')
flags.DEFINE_integer('k', 8, 'Variety count of motifs graph.')
flags.DEFINE_float('train_rate', 0.8, 'Rate of training set.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_string('model_name', 'EST-RVI', 'Model name')

# Retrieve flags
model_name = FLAGS.model_name
train_rate = FLAGS.train_rate
seq_len = FLAGS.seq_len
output_dim = FLAGS.pre_len
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
units = FLAGS.units
k=FLAGS.k

# Load data
data = np.load('./TrafficDataset/DynamicTraffic.npy')
adj = np.load('./TrafficDataset/Adj.npy')
time_len = data.shape[0]
num_nodes = data.shape[1]
max_value = np.max(data)

data_normalized = np.mat(data, dtype=np.float32) / np.max(data)  # Normalize data

# Multimodal data preparation
tqdm.pandas()
Sub_Edges = np.load('./EventDataset/DynamicEvents/subG_edges.npy')
subG = DirectedGraph().sub_graph(Sub_Edges)
adj = DirectedGraph().get_adjacency_matrix(subG)
way2id = DirectedGraph().get_edge_id_order(subG)
datatxt = pd.read_table("./EventDataset/DynamicEvents/AuxiliaryInfo", delimiter=",", header=None)
datatxt.columns = ["time", "wayid", 'xishu', 'width', 'waylen', 'lanenum', 'quary_count', 'precipitation1', 'Wind_Speed']

# Static features data
static_f = ['width', 'waylen', 'lanenum']
static_featureNum = len(static_f)
static_features = np.zeros((num_nodes, static_featureNum))
for row in tqdm(datatxt.itertuples()):
    idx = way2id[row.wayid]
    static_features[idx, 0] = row.width
    static_features[idx, 1] = row.waylen
    static_features[idx, 2] = row.lanenum

# Normalize static features
for j in range(static_features.shape[1]):
    static_features_max = static_features[:, j].max(axis=0, keepdims=True)
    static_features[:, j] = static_features[:, j] / static_features_max

# Quadruple form data
varis_f = ['quary_count', 'precipitation1', 'wind_speed']
varis_f_num = len(varis_f)
weather_map = get_weather_map()  # Assuming get_weather_map() is defined elsewhere

times_inp = np.zeros((time_len, varis_f_num), dtype='float32')
spatial_inp = np.zeros((time_len, varis_f_num), dtype='float32')
values_inp = np.zeros((time_len, varis_f_num), dtype='float32')
varis_inp = np.zeros((time_len, varis_f_num), dtype='int32')

for row in tqdm(datatxt.itertuples()):
    if float(row.xishu) > 50:
        times_inp[row.time, 0] = row.time
        spatial_inp[row.time, 0] = way2id[row.wayid]
        values_inp[row.time, 0] = row.quary_count
        varis_inp[row.time, 0] = 1

    precipitation1, wind_speed = weather_map[int(row.time) % 4].split(",")
    if float(precipitation1) > 100:
        times_inp[row.time, 1] = row.time
        spatial_inp[row.time, 1] = way2id[row.wayid]
        values_inp[row.time, 1] = precipitation1
        varis_inp[row.time, 1] = 2
    if float(wind_speed) > 25:
        times_inp[row.time, 2] = row.time
        spatial_inp[row.time, 2] = way2id[row.wayid]
        values_inp[row.time, 2] = wind_speed
        varis_inp[row.time, 2] = 3

# Prepare data for training
trainX, trainY, testX, testY = preprocess_data(data_normalized, static_features, times_inp, spatial_inp, values_inp,
                                               varis_inp, time_len, train_rate, seq_len, output_dim)

# Calculate total batches
totalbatch = int(trainX.shape[0] / batch_size)
totaltestbatch = int(testX.shape[0] / batch_size)


def motif_gcn(inputs, _weights, _biases):
    cell_1 = TIACell(units, adj,k, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    inputs = tf.unstack(inputs, axis=1)
    outputs, states = tf.nn.static_rnn(cell, inputs, dtype=tf.float32)

    motifs = []
    for output in outputs:
        reshaped_output = tf.reshape(output, shape=[-1, num_nodes, units])
        reshaped_output = tf.reshape(reshaped_output, shape=[-1, units])
        motifs.append(reshaped_output)

    last_output = motifs[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output, shape=[-1, num_nodes, output_dim])
    output = tf.transpose(output, perm=[0, 2, 1])
    output = tf.reshape(output, shape=[-1, num_nodes])

    return output, motifs, states


# Placeholders
inputs = tf.placeholder(tf.float32, shape=[32, seq_len, num_nodes + 3 * 4])
labels = tf.placeholder(tf.float32, shape=[32, output_dim, num_nodes])

# Graph weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([units, output_dim], mean=1.0), name='weight_out')
}
biases = {
    'out': tf.Variable(tf.random_normal([output_dim]), name='bias_out')
}

# Build the model
predictions, motifs, final_states = motif_gcn(inputs, weights, biases)
predicted_values = predictions

# Loss function and optimizer
lambda_loss = 0.0015
regularization_loss = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
labels_flat = tf.reshape(labels, [-1, num_nodes])
loss = tf.reduce_mean(tf.nn.l2_loss(predicted_values - labels_flat) + regularization_loss)
error = tf.sqrt(tf.reduce_mean(tf.square(predicted_values - labels_flat)))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

# Initialize session
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())
#sess = tf.Session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.run(tf.global_variables_initializer())

# Define paths
output_directory = 'out/%s' % model_name
sub_directory = '%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r' % (
    model_name, lr, batch_size, units, seq_len, output_dim, training_epoch)
path = os.path.join(output_directory, sub_directory)

# Create directory if it doesn't exist
if not os.path.exists(path):
    os.makedirs(path)


# Define evaluation function
def evaluation(truev_subset, predv_subset, max_value):
    re_truev = truev_subset * max_value
    mask = (re_truev >= 50)
    truev = truev_subset[mask]
    predv = predv_subset[mask]

    rmse = math.sqrt(mean_squared_error(truev, predv))
    mae = mean_absolute_error(truev, predv)
    F_norm = la.norm(truev - predv) / la.norm(truev)
    mape = mean_absolute_percentage_error(truev, predv)

    return rmse, mae, 1 - F_norm, mape


# Training and testing loops
x_axe, batch_loss, batch_rmse, batch_pred = [], [], [], []
test_loss, test_rmse, test_mae, test_acc, test_mape, test_pred = [], [], [], [], [], []

max_acc = 0
for epoch in range(training_epoch):
    for m in range(totalbatch):
        mini_batch = trainX[m * batch_size: (m + 1) * batch_size]
        mini_label = trainY[m * batch_size: (m + 1) * batch_size]

        _, loss1, rmse1, train_output = sess.run([optimizer, loss, error, predicted_values],
                                                 feed_dict={inputs: mini_batch, labels: mini_label})
        batch_loss.append(loss1)
        batch_rmse.append(rmse1 * max_value)

    test_all = []
    test_pre_all = []
    for m in range(totaltestbatch):
        mini_batch = testX[m * batch_size: (m + 1) * batch_size]
        mini_label = testY[m * batch_size: (m + 1) * batch_size]

        loss2, rmse2, test_output = sess.run([loss, error, predicted_values],
                                             feed_dict={inputs: mini_batch, labels: mini_label})
        test_all.append(mini_label)
        test_pre_all.append(test_output)

    test_label = np.reshape(test_all, [-1, num_nodes])
    test_output = np.reshape(test_pre_all, [-1, num_nodes])

    rmse, mae, acc, mape = evaluation(test_label, test_output, max_value)
    test_label1 = test_label * max_value
    test_output1 = test_output * max_value

    test_loss.append(loss2)
    test_rmse.append(rmse)
    test_mae.append(mae)
    test_acc.append(acc)
    test_mape.append(mape)
    test_pred.append(test_output1)

    print('Iter:{}'.format(epoch),
          'test_mae:{:.7}'.format(mae),
          'test_rmse:{:.7}'.format(rmse),
          'test_mape:{:.7}'.format(mape),
          'test_acc:{:.7}'.format(acc))

    if max_acc < acc:
        max_acc = acc
        saver.save(sess, path + '/EST-RVI_Epoch_%r' % epoch, global_step=epoch)

time_end = time.time()
print(time_end - time_start, 's')

b = int(len(batch_rmse) / totalbatch)
batch_rmse1 = [i for i in batch_rmse]
train_rmse = [(sum(batch_rmse1[i * totalbatch:(i + 1) * totalbatch]) / totalbatch) for i in range(b)]

batch_loss1 = [i for i in batch_loss]
train_loss = [(sum(batch_loss1[i * totalbatch:(i + 1) * totalbatch]) / totalbatch) for i in range(b)]

index = test_acc.index(max(test_acc))
test_result = test_pred[index]

var = pd.DataFrame(test_result)
var.to_csv(path + '/predict_value.csv', index=False, header=False)

print('min_mae:%r' % (test_mae[index]),
      'min_rmse:%r' % (test_rmse[index]),
      'min_mape:%r' % (test_mape[index]),
      'max_acc:%r' % (test_acc[index]))

sess.close()