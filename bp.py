import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import preprocessing
from pandas.core.frame import DataFrame

# sess = tf.InteractiveSession()
path_train = 'data/train/train.csv'
path_test = 'data/test/test.csv'

data = pd.read_csv(path_train, low_memory=False)
data_test = pd.read_csv(path_test, low_memory=False)

X = data.values[:60000, 0:-2].tolist()
Y_label = data.values[:60000, -2].tolist()
user_id = data.values[:60000, -1].tolist()
label = []
for i in Y_label:
    if i not in label :
        label.append(i)
Y = []
for j in Y_label:
    for k in range(len(label)):
        if j == label[k]:
            Y.append(k)

#归一化
X = preprocessing.scale(X)

n_input = 25
n_output = 15
#学习率
learning_rate = 0.01
training_step = 100
batch_size = 100
n_hidden = 15
n_hidden2 = 10

#神经网络层定义

def inference(x_input):
    with tf.variable_scope("hidden"):
        weights = tf.get_variable("weights", [n_input, n_hidden], initializer=tf.random_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [n_hidden], initializer=tf.constant_initializer(0.0))
        hidden = tf.nn.relu(tf.nn.dropout(tf.matmul(x_input, weights) + biases,0.8))

    with tf.variable_scope("hidden2"):
        weights = tf.get_variable("weights", [n_hidden, n_hidden2], initializer=tf.random_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [n_hidden2], initializer=tf.constant_initializer(0.0))
        hidden2 = tf.nn.relu(tf.nn.dropout(tf.matmul(hidden, weights) + biases,0.8))

    with tf.variable_scope("out"):
        weights = tf.get_variable("weights", [n_hidden2, n_output], initializer=tf.random_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [n_output], initializer=tf.constant_initializer(0.0))
        output = tf.nn.softmax(tf.matmul(hidden2, weights) + biases)

    return output

# loss_all = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_label, name='cross_entropy_loss')
# loss = tf.reduce_mean(loss_all, name='avg_loss')

#入参25，出参15
x = tf.placeholder('float',[None,n_input])
y = tf.placeholder('float',[None,n_output])

#interface构建图，返回包含预测结果的tensor
pred = inference(x)

# 计算损失函数
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 1), logits=pred), name='avg_loss')
# 定义优化器，梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

#计算损失
# loss_all = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y, name='cross_entropy_loss')
# loss = tf.reduce_mean(loss_all, name='avg_loss')

# 定义准确率计算
#equal比较两个列表返回True False列表，argmax返回最大数列表值的下标
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#tf.cast转换列表的数据类型取平均得到正确率
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#初始化
init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    Y = tf.one_hot(Y,15).eval()  # 在使用t.eval()时，等价于：tf.get_default_session().run(t).
    X, Y = shuffle(X, Y)
    scaler = preprocessing.StandardScaler()

    X = scaler.fit_transform(X)

    x_train = X[0:40000]
    y_train = Y[0:40000]

    x_validate = X[40000:50000]
    y_validate = Y[40000:50000]

    x_test = X[50000:]
    y_test = Y[50000:]

    validate_data = {x:x_validate, y:y_validate}
    test_data = {x: x_test, y: y_test}
    for i in range(training_step):
        avg_loss = 0.
        total_batch = int(len(X) / batch_size)
        # xs,ys为每个batch_size的训练数据与对应的标签
        for j in range(total_batch):
            xs = x_train
            ys = y_train
            _, l = sess.run([optimizer, cross_entropy], feed_dict={x: xs, y: ys})
            avg_loss += l / total_batch
        # 每1000次训练打印一次损失值与验证准确率
        if i % 10 == 0:
            validate_accuracy = sess.run(accuracy, feed_dict=validate_data)
            print("after %d training steps, the loss is %g, the validation accuracy is %g" % (
                i, l, validate_accuracy))

    print("the training is finish!")
    # 最终的测试准确率

    # acc = sess.run(accuracy, feed_dict=test_data)

    X_test = data_test.values[:, 0:-1].tolist()
    user_id_test = data_test.values[:, -1].tolist()
    #归一化
    X_test = preprocessing.scale(X_test)
    #标准化
    scaler = preprocessing.StandardScaler()
    X_test = scaler.fit_transform(X_test)

    acc2 = sess.run(pred, feed_dict={x:X_test})
    index_test = np.argmax(acc2, axis=1)
    result = []
    for i in index_test:
        re_test = label[i]
        result.append(re_test)

    frame_pred = DataFrame(user_id_test)
    frame_pred['predict'] = result

    path3 = 'submit.csv'
    print("保存开始")
    frame_pred.rename(columns={0: 'user_id', 1: 'predict'}, inplace=True)
    frame_pred.to_csv(path3, index=False)
    print('保存结束')