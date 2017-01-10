#!/home/pedgrfx/anaconda3/bin/python

# Load pickled data
import pickle

# TODO: fill this in based on where you saved the training and testing data
training_file = './train.p'
testing_file = './test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
del train
del test

### To start off let's do a basic data summary.
import numpy as np

# TODO: number of training examples
n_train = X_train.shape[0]

# TODO: number of testing examples
n_test = X_test.shape[0]

# TODO: what's the shape of an image?
image_shape = X_train.shape[1:4]

# TODO: how many classes are in the dataset
n_classes = np.max(y_train) + 1  

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

import matplotlib.pyplot as plt

### Preprocess the data here.
### Feel free to use as many code cells as needed.
import cv2

# Let's convert the images to grayscale and mean normalize
X_train_inputList = []
for color_image in X_train:
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    avg_pix_rows = np.mean(gray_image, axis=0)
    avg_pix_val = np.mean(avg_pix_rows, axis=0)
    gray_image = gray_image - avg_pix_val
    gray_image = gray_image / 256.0
    X_train_inputList.append(gray_image)
X_train_inputs = np.array(X_train_inputList)
del X_train_inputList
del X_train

X_test_inputList = []
for color_image in X_test:
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    avg_pix_rows = np.mean(gray_image, axis=0)
    avg_pix_val = np.mean(avg_pix_rows, axis=0)
    gray_image = gray_image - avg_pix_val
    gray_image = gray_image / 256.0
    X_test_inputList.append(gray_image)
X_test_inputs = np.array(X_test_inputList)
del X_test_inputList
del X_test

### Generate data additional (if you want to!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.

#train_subset = 20000 # Use -1 for full dataset
train_subset = -1 # Use -1 for full dataset

# Randomize the dataset (NB make sure you do this before splitting into train and validation sets)
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
X_train_inputs, y_train = randomize(X_train_inputs, y_train)
X_test_inputs, y_test = randomize(X_test_inputs, y_test)

# Split train_dataset and labels into a training set and a CV set
train_set_size = int(len(y_train) * 3 / 4)
print("Training size : {}".format(train_set_size))
print("Validation size : {}".format(len(y_train) - train_set_size))
train_dataset = X_train_inputs[:train_set_size]
train_labels = y_train[:train_set_size]
valid_dataset = X_train_inputs[train_set_size:]
valid_labels = y_train[train_set_size:]
test_dataset = X_test_inputs
test_labels = y_test
del X_train_inputs
del y_train
del X_test_inputs
del y_test

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# Resize dataset to subset if needed
if train_subset == -1:
    train_subset = len(train_dataset)
train_dataset = train_dataset[:train_subset, :]
train_labels = train_labels[:train_subset]

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, 32, 32,1)).astype(np.float32)
  labels = (np.arange(n_classes) == labels[:,None]).astype(np.float32)
  return dataset, labels

# We need to rehape the image data into a (32,32,1) dimension to use as input to the CNN
# Also one-shot encode the labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf

# START OF MAIN PROGRAM
image_sizeX = 32
image_sizeY = 32
num_channels = 1 # grayscale
num_labels = 43  

epochs = 20
batch_size = 128
patch_size = 5
depth = 16
hidden_layer1_size =  1024
hidden_layer2_size =  512
dropout_keep_prob = 0.5
use_cnn = True
use_regularization = True
reg_beta = 0.01
use_learning_rate_decay = False
use_dropout = False
initial_learning_rate = 0.002

graph = tf.Graph()
with graph.as_default():
    
    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=[None, image_sizeX, image_sizeY, num_channels])
    tf_train_labels = tf.placeholder(tf.float32, shape=[None, num_labels])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
      
    # Variables.
    cnn_layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth]))
    cnn_layer1_weights = tf.get_variable("cnn_layer1_weights", 
                                            shape = [patch_size, patch_size, num_channels, depth],
                                            initializer=tf.contrib.layers.xavier_initializer())
    #cnn_layer1a_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth]))
    #cnn_layer1a_weights = tf.get_variable("cnn_layer1a_weights", 
    #                                        shape = [patch_size, patch_size, depth, depth],
    #                                        initializer=tf.contrib.layers.xavier_initializer())
    cnn_layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth]))
    cnn_layer2_weights = tf.get_variable("cnn_layer2_weights", 
                                        shape = [patch_size, patch_size, depth, depth],
                                        initializer=tf.contrib.layers.xavier_initializer())

    weights_fcnn_h1 = tf.Variable(tf.truncated_normal([image_sizeX * image_sizeY * num_channels, hidden_layer1_size]))
    weights_fcnn_h1 = tf.get_variable("weights_fcnn_h1", 
                                        shape = [image_sizeX * image_sizeY * num_channels, hidden_layer1_size], 
                                        initializer=tf.contrib.layers.xavier_initializer())
    weights_fcnn_h2 = tf.Variable(tf.truncated_normal([hidden_layer1_size, hidden_layer2_size]))
    weights_fcnn_h2 = tf.get_variable("weights_fcnn_h2", 
                                        shape = [hidden_layer1_size, hidden_layer2_size], 
                                        initializer=tf.contrib.layers.xavier_initializer())
    weights_fcnn_o = tf.Variable(tf.truncated_normal([hidden_layer2_size, num_labels]))
    weights_fcnn_o = tf.get_variable("weights_fcnn_o", 
                                        shape = [hidden_layer2_size, num_labels], 
                                        initializer=tf.contrib.layers.xavier_initializer())


    cnn_layer1_biases = tf.Variable(tf.zeros([depth]))
    #cnn_layer1a_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    cnn_layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    biases_fcnn_h1 = tf.Variable(tf.constant(1.0, shape=[hidden_layer1_size]))
    biases_fcnn_h2 = tf.Variable(tf.constant(1.0, shape=[hidden_layer2_size]))
    biases_fcnn_o = tf.Variable(tf.constant(1.0, shape=[num_labels]))
        
    keep_prob = tf.placeholder(tf.float32)

    # CNN piece
    def cnn_model(data):
        conv1 = tf.nn.conv2d(data, cnn_layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden1 = tf.nn.relu(conv1 + cnn_layer1_biases)
        #conv1a = tf.nn.conv2d(hidden1, cnn_layer1a_weights, [1, 1, 1, 1], padding='SAME')
        #hidden1a = tf.nn.relu(conv1a + cnn_layer1a_biases)
        pooling1 = tf.nn.max_pool(hidden1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        conv2 = tf.nn.conv2d(pooling1, cnn_layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden2 = tf.nn.relu(conv2 + cnn_layer2_biases)
        pooling2 = tf.nn.max_pool(hidden2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        shape = pooling2.get_shape().as_list()
        #return tf.reshape(pooling2, [shape[0], shape[1] * shape[2] * shape[3]])
        dim = np.prod(shape[1:])
        return tf.reshape(pooling2, [-1, dim])
    
    # FC_NN piece
    def fcnn_model(data):
        hidden1_fcnn = tf.nn.relu(tf.matmul(data, weights_fcnn_h1) + biases_fcnn_h1)
        if use_dropout:
            hidden1_drop_fcnn = tf.nn.dropout(hidden1_fcnn, keep_prob)
        else:
            hidden1_drop_fcnn = hidden1_fcnn
        hidden2_fcnn = tf.nn.relu(tf.matmul(hidden1_drop_fcnn, weights_fcnn_h2) + biases_fcnn_h2)
        if use_dropout:
            hidden2_drop_fcnn = tf.nn.dropout(hidden2_fcnn, keep_prob)
        else:
            hidden2_drop_fcnn = hidden2_fcnn
        return tf.matmul(hidden2_drop_fcnn, weights_fcnn_o) + biases_fcnn_o
    
    # Full network
    def full_model(data):   
        cnn_data = cnn_model(data)
        pred = fcnn_model(cnn_data)
        return pred

    # Instanciate the model
    logits_pred = full_model(tf_train_dataset)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits_pred, tf_train_labels)

    if use_regularization:
        loss += reg_beta*tf.nn.l2_loss(cnn_layer1_weights) + \
                reg_beta*tf.nn.l2_loss(cnn_layer2_weights) + \
                reg_beta*tf.nn.l2_loss(weights_fcnn_h1) + \
                reg_beta*tf.nn.l2_loss(weights_fcnn_h2) + \
                reg_beta*tf.nn.l2_loss(weights_fcnn_o) 
                #reg_beta*tf.nn.l2_loss(cnn_layer1a_weights) + 

    loss = tf.reduce_mean(loss)
    
    # Optimizer.
    optimizer = tf.train.AdamOptimizer(initial_learning_rate).minimize(loss)
      
    # Predictions for the training, validation, and test data.
    train_predictions = tf.nn.softmax(logits_pred)
    valid_predictions = full_model(tf_valid_dataset)       
    test_predictions = full_model(tf_test_dataset)
    
    def accuracy(predictions, labels):
        return 1.0 * (np.sum(np.argmax(predictions, 1) == np.argmax(labels,1)) / predictions.shape[0])

    ### Train your model here.
    ### Feel free to use as many code cells as needed.
    
    import math
    from tqdm import tqdm
    import time

    log_batch_step = 100
    
    # Train the model        
    start = time.time()
    batches = []
    loss_batch = []
    train_acc_batch = []
    valid_acc_batch = []

    with tf.Session(graph=graph) as session:
        # This is a one-time operation which ensures the parameters get initialized as
        # we described in the graph: random weights for the matrix, zeros for the biases. 
        tf.initialize_all_variables().run()
        batch_count = int(math.ceil(len(train_dataset)/batch_size))

        for epoch_i in range(epochs):
            # Progress bar
            batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')

            # The training cycle
            for batch_i in batches_pbar:
                # Generate a minibatch.
                batch_start = batch_i * batch_size
                batch_data = train_dataset[batch_start:batch_start + batch_size, :]
                batch_labels = train_labels[batch_start:batch_start + batch_size, :]
                # Prepare a dictionary telling the session where to feed the minibatch.
                feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: dropout_keep_prob}
                # Run the optimizer adn get loss and predictions
                #print("batch_i {} batch_data.shape {} batch_labels.shape {}".format(batch_i, batch_data.shape, batch_labels.shape))
                _, l, prediction = session.run([optimizer, loss, train_predictions], feed_dict=feed_dict)
                # Log every certain number of batches
                if not batch_i % log_batch_step:
                    batch_acc = accuracy(prediction, batch_labels)
                    valid_acc = accuracy(session.run(valid_predictions,feed_dict={keep_prob:1.0}), valid_labels)
                    #print("prediction[0] : {}".format(prediction[0]))
                    #print("labels[0] : {}".format(batch_labels[0]))
                    #print("prediction : {}".format(prediction))
                    #print("labels : {}".format(batch_labels))
                    print("batch_acc {}".format(batch_acc))
                    print("valid_acc {}".format(valid_acc))
                    # Log batches
                    previous_batch = batches[-1] if batches else 0
                    batches.append(log_batch_step + previous_batch)
                    loss_batch.append(l)
                    train_acc_batch.append(batch_acc)
                    valid_acc_batch.append(valid_acc)                
            print("Loss after epoch {} is {}".format(epoch_i+1, l))

        # Check model accuracy against test set
        test_acc = accuracy(session.run(test_predictions, feed_dict={keep_prob:1.0}), test_labels)

        print('Test accuracy: {}'.format(test_acc))
        end = time.time()
        print("Time taken to train database : {} seconds".format(end - start))

        # Print graphs
        loss_plot = plt.subplot(211)
        loss_plot.set_title('Loss')
        loss_plot.plot(batches, loss_batch, 'g')
        loss_plot.set_xlim([batches[0], batches[-1]])
        acc_plot = plt.subplot(212)
        acc_plot.set_title('Accuracy')
        acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
        acc_plot.plot(batches, valid_acc_batch, 'b', label='Validation Accuracy')
        acc_plot.set_ylim([0, 1.0])
        acc_plot.set_xlim([batches[0], batches[-1]])
        acc_plot.legend(loc=4)
        plt.tight_layout()
        plt.show()
