import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

XDim=864
YDim=512
ZDim=4

XStokes=875
YStokes=512
ZStokes=4

XMagfld=875
YMagfld=512
ZMagfld=3

pathTrain = '../data/train.tfr'  # The TFRecord file containing the training set
pathValid = '../data/val.tfr'    # The TFRecord file containing the validation set
pathTest = '../data/test.tfr'    # The TFRecord file containing the test set

batchSize=6
batchN=1

with tf.Session() as sess:
    feature = {
        'magfld': tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True),
        'stokes': tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True)
      }
        
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([pathTrain], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    """Parse a single record into x and y images"""
    x = features['stokes']
    x = tf.reshape(x, (YStokes, XStokes, ZStokes))
    x = tf.slice(x, (0, 0, 0), (YDim, XDim, ZStokes))
    
    y = features['magfld']
    y = tf.reshape(y, (YMagfld, XMagfld, ZMagfld))
    y = tf.slice(y, (0, 0, 0), (YDim, XDim, ZMagfld))
    #y = tf.cast(features['train/period'], tf.float32)
    
    # Any preprocessing here ...
    
    # Creates batches by randomly shuffling tensors
    X, Y = tf.train.batch([x, y], batch_size=batchSize)

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Now we read batches of images and labels and plot them
    for batch_index in range(batchN):
        level1, level2 = sess.run([X, Y])
        for i in range(X.shape[0]):
            imLevel1 = level1[i,0:512,0:864,0]
            plt.gray()
            plt.imshow(imLevel1)
            plt.show()
            imLevel2 = level2[i, 0:512,0:864,0]
            plt.imshow(imLevel2)
            plt.show()


    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()
