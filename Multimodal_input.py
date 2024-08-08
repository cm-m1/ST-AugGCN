import numpy as np

def preprocess_data(data,static_features,times_inp,spatial_inp,values_inp, varis_inp,time_len, train_rate, seq_len, pre_len):
    train_size = int(time_len * train_rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]

    spatial_inp_train = spatial_inp[0:train_size]
    spatial_inp_test = spatial_inp[train_size:time_len]

    times_inp_train = times_inp[0:train_size]
    times_inp_test = times_inp[train_size:time_len]

    varis_inp_train = varis_inp[0:train_size]
    varis_inp_test = varis_inp[train_size:time_len]

    values_inp_max = np.max(np.max(values_inp))
    values_inp = values_inp / values_inp_max
    values_inp_train = values_inp[0:train_size]
    values_inp_test = values_inp[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a1 = train_data[i: i + seq_len + pre_len]
        a2 = varis_inp_train[i: i + seq_len + pre_len]
        a3= times_inp_train[i: i + seq_len + pre_len]
        a4 = values_inp_train[i: i + seq_len + pre_len]
        a5=spatial_inp_train[i: i + seq_len + pre_len]
        a =np.column_stack((a1[0:seq_len], a2[0: seq_len],a3[0: seq_len],a4[0: seq_len],a5[0: seq_len]))
        trainX.append(a)
        trainY.append(a1[seq_len: seq_len + pre_len])
    for i in range(len(test_data) - seq_len - pre_len):
        b1 = test_data[i: i + seq_len + pre_len]
        b2 = varis_inp_test[i: i + seq_len + pre_len]
        b3= times_inp_test[i: i + seq_len + pre_len]
        b4 = values_inp_test[i: i + seq_len + pre_len]
        b5 = spatial_inp_test[i: i + seq_len + pre_len]
        b = np.column_stack((b1[0:seq_len], b2[0: seq_len], b3[0: seq_len], b4[0: seq_len],b5[0: seq_len]))
        testX.append(b)
        testY.append(b1[seq_len: seq_len + pre_len])

    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)

    return trainX1, trainY1, testX1, testY1