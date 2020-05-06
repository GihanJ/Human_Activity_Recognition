#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd   # import data
import numpy as np    # to perform numerical functions
import pickle         # serialize objects
import matplotlib.pyplot as plt # for plotting
from scipy import stats     # for statistical features
import tensorflow as tf     # as the backend to create RNN
import seaborn as sns       # beautifying graphs
from pylab import rcParams
from sklearn import metrics  # to judge the model
from sklearn.model_selection import train_test_split  # to split and training purposes


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # to ignore memory allocation exceeds warnings

sns.set(style='whitegrid', palette='muted', font_scale=1.5) # beautify the graphs

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42    # initializing random seed

columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']     # defining columns of the input data
df = pd.read_csv('data/WISDM_ar_v1.1_raw.txt', header = None, names = columns)  # read data
df = df.dropna()    # dropping records which have missing columns

df.head()   # display starting records of data
df.info()   # display a summary of loaded data frame

df['activity'].value_counts().plot(kind='bar', title='Training examples by activity type');
df['user'].value_counts().plot(kind='bar', title='Training examples by user');

# plot activities by x,y,z
def plot_activity(activity, df):
    data = df[df['activity'] == activity][['x-axis', 'y-axis', 'z-axis']][:200]
    axis = data.plot(subplots=True, figsize=(16, 12),
                     title=activity)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))

plot_activity("Sitting", df)
plot_activity("Standing", df)
plot_activity("Walking", df)
plot_activity("Jogging", df)
plot_activity("Downstairs", df)
plot_activity("Upstairs", df)

# separating data to blocks and labelling to feed into LSTM network
N_TIME_STEPS = 200
N_FEATURES = 3
step = 20
segments = []
labels = []
for i in range(0, len(df) - N_TIME_STEPS, step):
    xs = df['x-axis'].values[i: i + N_TIME_STEPS]
    ys = df['y-axis'].values[i: i + N_TIME_STEPS]
    zs = df['z-axis'].values[i: i + N_TIME_STEPS]
    label = stats.mode(df['activity'][i: i + N_TIME_STEPS])[0][0]
    segments.append([xs, ys, zs])
    labels.append(label)


print("Array (non shaped) Segments Shape :", np.array(segments).shape)  # checking the data shape


reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)  # labelling activities to a float version

print("Reshaped Segments Shape :", reshaped_segments.shape)


X_train, X_test, y_train, y_test = train_test_split(
        reshaped_segments, labels, test_size=0.2, random_state=RANDOM_SEED)

print("X_train Length :", len(X_train))

print("X_test Length :", len(X_test))

# Building the model
N_CLASSES = 6
N_HIDDEN_UNITS = 64 # unit size for each LSTM layer

def create_LSTM_model(inputs):
    W = {
        'hidden': tf.Variable(tf.random_normal([N_FEATURES, N_HIDDEN_UNITS])),
        'output': tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN_UNITS], mean=1.0)),
        'output': tf.Variable(tf.random_normal([N_CLASSES]))
    }

    X = tf.transpose(inputs, [1, 0, 2])
    X = tf.reshape(X, [-1, N_FEATURES]) # transforming and reshaping to feed into th model
    hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])   # using relu as the activation function for the hidden layer - this is the LSTM main function
    hidden = tf.split(hidden, N_TIME_STEPS, 0) # splitting data to 200

    # Stack 2 LSTM layers
    lstm_layers = [tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)] # creating 2 lstm layers
    lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers) # stacking layers

    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32) # creating the LSTM network from 2 layers

    # return the output of the whole RNN
    lstm_last_output = outputs[-1]

    return tf.matmul(lstm_last_output, W['output']) + biases['output']


tf.reset_default_graph()
# feed data to tensorflow model as x and Y
X = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name="input")
Y = tf.placeholder(tf.float32, [None, N_CLASSES])


pred_Y = create_LSTM_model(X)  # calling the function

pred_softmax = tf.nn.softmax(pred_Y, name="y_")   # softmax as the activation function for the output layer


L2_LOSS = 0.0015  # to prevent from over fitting

l2 = L2_LOSS * \
     sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred_Y, labels = Y)) + l2



LEARNING_RATE = 0.0025

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss) # setting adam optimizer for learning

correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))  # finding the highest probability prediction
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

N_EPOCHS = 50
BATCH_SIZE = 1024

saver = tf.train.Saver()  # to save the training process in disk

# saving everything in a dictionary to use to plotting and perform analysis of the network
history = dict(train_loss=[],
                     train_acc=[],
                     test_loss=[],
                     test_acc=[])

sess=tf.InteractiveSession() # create a session to start with training the LSTM model
sess.run(tf.global_variables_initializer())

train_count = len(X_train)

# starting the training process
for i in range(1, N_EPOCHS + 1):
    for start, end in zip(range(0, train_count, BATCH_SIZE),
                          range(BATCH_SIZE, train_count + 1,BATCH_SIZE)):
        sess.run(optimizer, feed_dict={X: X_train[start:end],
                                       Y: y_train[start:end]})

    _, acc_train, loss_train = sess.run([pred_softmax, accuracy, loss], feed_dict={
                                            X: X_train, Y: y_train})

    _, acc_test, loss_test = sess.run([pred_softmax, accuracy, loss], feed_dict={
                                            X: X_test, Y: y_test})

    history['train_loss'].append(loss_train)
    history['train_acc'].append(acc_train)
    history['test_loss'].append(loss_test)
    history['test_acc'].append(acc_test)

    if i != 1 and i % 10 != 0:
        continue
    print(f'epoch: {i} test accuracy: {acc_test} loss: {loss_test}')

predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={X: X_test, Y: y_test})

print()
print(f'final results: accuracy: {acc_final} loss: {loss_final}')



pickle.dump(predictions, open("predictions.p", "wb"))
pickle.dump(history, open("history.p", "wb"))
tf.train.write_graph(sess.graph_def, '.', './checkpoint/har.pbtxt')
saver.save(sess, save_path = "./checkpoint/har.ckpt")
sess.close()


history = pickle.load(open("history.p", "rb"))
predictions = pickle.load(open("predictions.p", "rb"))


plt.figure(figsize=(12, 8))

plt.plot(np.array(history['train_loss']), "r--", label="Train loss")
plt.plot(np.array(history['train_acc']), "g--", label="Train accuracy")

plt.plot(np.array(history['test_loss']), "r-", label="Test loss")
plt.plot(np.array(history['test_acc']), "g-", label="Test accuracy")

plt.title("Training session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress (Loss or Accuracy values)')
plt.xlabel('Training Epoch')
plt.ylim(0)

plt.show()

# confusion matrix
LABELS = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']


max_test = np.argmax(y_test, axis=1)
max_predictions = np.argmax(predictions, axis=1)
confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)

plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# exporting the model to use in android app
from tensorflow.python.tools import freeze_graph

MODEL_NAME = 'har'

input_graph_path = 'checkpoint/' + MODEL_NAME+'.pbtxt'
checkpoint_path = './checkpoint/' +MODEL_NAME+'.ckpt'
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'

freeze_graph.freeze_graph(input_graph_path, input_saver="",
                          input_binary=False, input_checkpoint=checkpoint_path,
                          output_node_names="y_", restore_op_name="save/restore_all",
                          filename_tensor_name="save/Const:0",
                          output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes="")


