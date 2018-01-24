
"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import os

import numpy as np
import tensorflow as tf 

from tensorflow.contrib import learn
from alexnet import AlexNet
from devisenet import DeViSENet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
from sklearn import preprocessing
from tensorflow.contrib.tensorboard.plugins import projector

"""
Configuration Part.
"""

# Path to the textfiles for the trainings and validation set
train_file = './train03.txt'
val_file = './test02.txt'
test_file = './val02.txt'

#global_step = tf.Variable(0, trainable=False)
#learning_rate = tf.train.exponential_decay(0.01, global_step,100,0.1,staircase=True)

# Learning params
learning_rate = 0.001 #0.01
num_epochs = 100
batch_size = 64

# Network params
dropout_rate = 0.5
num_classes = 100
train_layers = ['fc8', 'fc7', 'fc6'] # fc8 7 6

#train_layers_more = ['fc10','fc9','fc8', 'fc7', 'fc6']
# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "./tmp/DeViSE/tensorboard"
checkpoint_path = "./tmp/DeViSE/checkpoints"
checkpoint_path0 = "./tmp/finetune_alexnet/checkpoints"
"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='inference',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)
    tst_data = ImageDataGenerator(test_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=True)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)
testing_init_op = iterator.make_initializer(tst_data.data)

print('Start!')

label_list = [['' for i in range(3)] for j in range(num_classes)]
with open('labels0.txt', 'r') as f:
    i = 0
    for line in f.readlines():
        k = 0
        for word_line in line.strip().split(' '):
            label_list[i][k] = word_line
            k+=1
        i += 1


filename = 'glove.42B.300d.txt'
def loadGloVe(filename):
    w2v = {}
    vocab = ''
    embd = []
    #w2v_embedding = []
    #w2v_word = ['']
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab = row[0]
        embd = list(map(float, row[1:]))
        w2v[vocab] = embd
        #w2v_embedding.append(embd)
        #w2v_word.append(vocab)
    print('Loaded GloVe!')
    file.close()
    w2v_dim = len(embd)
    return w2v, w2v_dim

w2v, w2v_dim= loadGloVe(filename)

def get_normalize_w2v(k):
     t_vec = np.transpose(label_w2v[k]) 
     return t_vec / np.sqrt(np.sum(t_vec ** 2))


label_w2v = [[0 for i in range(w2v_dim)] for j in range(num_classes)]
for i in range(len(label_list)):
    num = 0
    l = [0 for k in range(w2v_dim)]
    tmp = np.array(l , dtype = np.float32)
    for j in range(len(label_list[i])):
        if label_list[i][j] in w2v:
            tmp = np.add(tmp ,w2v[label_list[i][j]])
            num +=1
    if num is not 0 :
        label_w2v[i] = tmp / num
        #w2v_word.append(label_list[i])
        #w2v_embedding.append(label_w2v[i])

#label_w2v_embedding = [[0 for i in range(w2v_dim)] for j in range(num_classes)]
#for k in range(len(label_w2v)):
#    label_w2v_embedding[k] = label_w2v[k]
#    label_w2v[k] = get_normalize_w2v(k)
def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

#labels_vec = tf.placeholder(tf.float32, [num_classes, w2v_dim])

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

print('Load Alexnet!')
# Link variable to model output
score = model.fc8

saver0 = tf.train.Saver()

#fc9 = fc(fc8, num_classes, num_classes, name='fc9')

#score = fc(fc9,num_classes,num_classes,name='fc10')

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))
##############################################################
#learning_rate = tf.placeholder(tf.float32)

t_label = tf.placeholder(tf.float32, [batch_size, w2v_dim])
t_j = tf.placeholder(tf.float32, [batch_size, num_classes, w2v_dim])
w2v_embd =  tf.placeholder(tf.float32,[num_classes,w2v_dim])

devise_model = DeViSENet(score, w2v_dim)
M = devise_model.M

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]


def get_max_margin(index):
    margin = 0.1
    max_margin = 0
   # score1 = tf.transpose(tf.matmul(tf.reshape(t_label[index], [1, 300]),M))
    score1 = tf.matmul(tf.reshape(t_label[index],[1,300]),tf.reshape(M[index],[300,1]))
    ###
    #tt = tf.reshape(t_label[index],[1,300])
    #t_norm = tf.sqrt(tf.reduce_sum(tt * tt, 1))
    #MM = tf.reshape(M[index], [300, 1])
    #M_norm = tf.sqrt(tf.reduce_sum(MM * MM, 1))
    #mul_norm = tf.reduce_sum(tt * MM, 1)
    #cos_sim = tf.div(mul_norm, t_norm * M_norm +1e-8) 
    for i in range(t_j.shape[1]):
        #score2 = tf.matmul(tf.reshape(t_j[index, i, :], [1, 300]), M))
        score2 = tf.matmul(tf.reshape(t_j[index, i, :],[1,300]),tf.reshape(M[index],[300,1]))
        max_margin += tf.maximum(0.0, margin - score1 + score2)
    return max_margin

with tf.name_scope("max_margin"):
    margin_loss = 0
    score1s = 0
    for i in range(batch_size):
        loss = get_max_margin(i)
        margin_loss += loss
        #score1s += score1
    margin_loss = tf.reduce_mean(margin_loss / batch_size)
    #score1s = tf.reduce_mean(score1s / batch_size)

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(margin_loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.AdagradOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)
#    train_op = optimizer.minimize(margin_loss)
with tf.name_scope("accuracy"):
    y_ = tf.argmax(y,1)
    similarity = tf.matmul(M,tf.transpose(w2v_embd))
    #top_k = 8
    #correct_num = 0
    #for i in range(batch_size):
    #    nearest = tf.nn.top_k(similarity[i],9)[1][1:]
    #    for k in range(top_k):
    #        if y_ == nearest[k] : 
    #            correct_num += 1
    #accuracy = tf.reduce_mean(correct_num / batch_size)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('max_margin_loss', margin_loss)
#tf.summary.scalar('accuracy',accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

saver = tf.train.Saver()

#summary_writer = tf.train.SummaryWriter(checkpoint_path)

# Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
#config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
#embedding = config.embeddings.add()
#embedding.tensor_name = embedding_var.name
# Link this tensor to its metadata file (e.g. labels).
#embedding.metadata_path = os.path.join(checkpoint_path , 'metadata.tsv')

############################################################3

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)
   # projector.visualize_embeddings(summary_writer, config)
    # To continue training from one of your checkpoints
    saver0.restore(sess, checkpoint_path0 + "/model_epoch200.ckpt")
    
    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        sess.run(training_init_op)
        test_count = 0
        loss = 0.
        total_loss = 0.
        total_acc = 0
        #y_score = 0.
        #total_score = 0
   
        for step in range(train_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)

            
            
            y_label = sess.run(tf.argmax(label_batch, 1))

            t_label_batch = np.zeros(shape=(batch_size, w2v_dim))
            t_j_batch = np.zeros(shape=(batch_size, num_classes, w2v_dim))
            for i in range(batch_size):
                for j in range(len(label_list)):
                    if j == y_label[i]:
                        t_label_batch[i]= get_normalize_w2v(j)
                    else:              
                        t_j_batch[i][j] = get_normalize_w2v(j)


            loss, _= sess.run([margin_loss, train_op], 
                                             feed_dict={x: img_batch,
                                             y: label_batch,
                                             keep_prob: 1.,
                                             t_label: t_label_batch,
                                             t_j: t_j_batch,
                                             w2v_embd:label_w2v})
#            if step % display_step == 0:
#                print('loss', loss)
#                s = sess.run(merged_summary, feed_dict={x: img_batch,
#                                                        y: label_batch,
#                                                        keep_prob: 1.,
#                                                        t_label: t_label_batch,
#                                                        t_j: t_j_batch,
#                                                        w2v_embd : label_w2v})
#                writer.add_summary(s, epoch*train_batches_per_epoch + step)
            
            total_loss += loss
            #total_score += y_score1
            test_count += 1
#        print("{} Start validation".format(datetime.now()))
#        sess.run(validation_init_op)
#        for _ in range(val_batches_per_epoch):
            #if step % 100 == 0:  
            sim = sess.run(similarity, feed_dict={x: img_batch,
	        					y: label_batch,
	        					keep_prob: 1.,
	        					t_label: t_label_batch,
	        			           	t_j: t_j_batch,
	        					w2v_embd : label_w2v})
            top_k = 8
            correct_num = 0
            for i in range(batch_size):
           #     print('------------ %d/64 batch ----------------' % (i))
           #     print(label_list[y_label[i]])
                nearest = (-sim[i]).argsort()[1:top_k+1]
                for k in range(top_k):
                    closed_word = label_list[nearest[k]]
                    if closed_word == label_list[y_label[i]]:
                        correct_num += 1
           #         log_str = '%d | Nearest word is %s' % (correct_num,closed_word)
           #         print(log_str)
            accuracy = correct_num / batch_size
           
           # acc = sess.run(accuracy, feed_dict={x: img_batch,
           #                                             y: label_batch,
           #                                             keep_prob: 1.,
           #                                             t_label: t_label_batch,
           #                                             t_j: t_j_batch,
           #                                             w2v_embd : label_w2v})
            if step % display_step == 0:
             
                print('loss', loss)
                print('accuracy: {}'.format(accuracy))
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.,
                                                        t_label: t_label_batch,
                                                        t_j: t_j_batch,
                                                        w2v_embd : label_w2v})
                writer.add_summary(s, epoch*train_batches_per_epoch + step)

            total_acc += accuracy
  
        #total_score /= test_count 
        total_acc /= test_count
        total_loss /= test_count
        print("{} Training Loss = {:.4f}".format(datetime.now(),
                                                       total_loss))
        print("{} Training Accuracy = {:.4f}".format(datetime.now(),
                                                       total_acc))       
        
        print("{} Saving checkpoint of model...".format(datetime.now()))
         
        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
