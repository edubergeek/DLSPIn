import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.optimizers import Adam

Version='v3-1'
XDim=864
YDim=512
ZDim=4
WDim=15

XInv=864
YInv=512
ZInv=4
WInv=15

XStokes=875
YStokes=512
ZStokes=4
WStokes=9

XMagfld=875
YMagfld=512
ZMagfld=3

pathTrain = './data/train.tfr'  # The TFRecord file containing the training set
pathValid = './data/val.tfr'    # The TFRecord file containing the validation set
pathTest = './data/test.tfr'    # The TFRecord file containing the test set
pathWeight = './data/%s.h5'%(Version)  # The HDF5 weight file generated for the trained model
pathModel = './data/%s.nn'%(Version)   # The model saved as a JSON file

batchSize=2
batchN=45

SPName = ['I', 'Q', 'U', 'V' ]
MagName = ['Strength', 'Inclination', 'Azimuth']
line=6302.8

def showInversion(name, imLevel1I, imLevel1Q, imLevel1U, imLevel1V, imLevel2S, imLevel2I, imLevel2A, imPredictS, imPredictI, imPredictA):
    # prepare to visualize
    plt.gray()
    fig1 = plt.figure(1)
    #fig1.suptitle('%s Level 1'%(name), fontsize=14)
    fig1.suptitle('%s Level 1'%(name), fontsize=14)
    plt.subplot(121)
    plt.title('%.2fA SP=%s'%(line, SPName[0]))
    plt.imshow(imLevel1I)
    plt.subplot(122)
    plt.title('%.2fA SP=%s'%(line, SPName[3]))
    plt.imshow(imLevel1V)
    fig2 = plt.figure(2)
    fig2.suptitle('%s Mag Field'%(name), fontsize=14)
    plt.subplot(121)
    plt.title('Level2 MF=%s'%(MagName[0]))
    plt.imshow(imLevel2S)
    plt.subplot(122)
    plt.title('Model MF=%s'%(MagName[0]))
    plt.imshow(imPredictS)
    fig3 = plt.figure(3)
    fig3.suptitle('%s Mag Field'%(name), fontsize=14)
    plt.subplot(121)
    plt.title('Level2 MF=%s'%(MagName[1]))
    plt.imshow(imLevel2I)
    plt.subplot(122)
    plt.title('Model MF=%s'%(MagName[1]))
    plt.imshow(imPredictI)
    fig4 = plt.figure(4)
    fig4.suptitle('%s Mag Field'%(name), fontsize=14)
    plt.subplot(121)
    plt.title('Level2 MF=%s'%(MagName[2]))
    plt.imshow(imLevel2A)
    plt.subplot(122)
    plt.title('Model MF=%s'%(MagName[2]))
    plt.imshow(imPredictA)
    plt.subplots_adjust(hspace=0.30, wspace=0.15)
    plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)


with tf.Session() as sess:
    feature = {
        'magfld': tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True),
        'stokes': tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True),
        'name':   tf.FixedLenFeature(shape=[], dtype=tf.string)
      }
        
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([pathTest], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    """Parse a single record into x and y images"""
    x = features['stokes']
    # unroll into a 4D array
    x = tf.reshape(x, (WStokes, YStokes, XStokes, ZStokes))
    # use slice to crop the data to the model dimensions - powers of 2 are a factor
    x = tf.slice(x, (0, 0, 0, 0), (WStokes, YDim, XDim, ZStokes))
    
    y = features['magfld']
    # unroll into a 3D array
    y = tf.reshape(y, (YMagfld, XMagfld, ZMagfld))
    # use slice to crop the data
    y = tf.slice(y, (0, 0, 0), (YDim, XDim, ZMagfld))
    #y = tf.cast(features['train/period'], tf.float32)
    
    name = features['name']

    # Any preprocessing here ...
    
    # Creates batches by randomly shuffling tensors
    X, Y, Name = tf.train.batch([x, y, name], batch_size=batchSize)

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)


    # main
    # load the trained model
    # load from YAML and create model
    model_file = open(pathModel, 'r')
    model_serial = model_file.read()
    model_file.close()
    model = model_from_yaml(model_serial)
    # load weights into new model
    model.load_weights(pathWeight)
    print("Loaded model from disk")
    model.summary()
 
    # evaluate loaded model on test data
    model.compile(optimizer=Adam(lr=1e-5), loss='mse', metrics=['mse'])

    WL = int(np.ceil(WInv/2))

    # Now we read batches of images and labels and plot them
    for batch_index in range(batchN):
        level1, level2, fname = sess.run([X, Y, Name])
        # forward pass through the model to invert the Level 1 SP images and predict the Level 2 image mag fld maps
        inversion = model.predict(level1, batchSize, verbose=1)

        # for each sample in the batch
        for i in range(X.shape[0]):
            #batchIn = np.reshape(image[0:WInv,0:YInv,0:XInv,0:ZInv], (1, WInv, YInv, XInv, ZInv))
            imPredictS = inversion[i,0:YInv,0:XInv,0]
            imPredictI = inversion[i,0:YInv,0:XInv,1]
            imPredictA = inversion[i,0:YInv,0:XInv,2]

            # visualize using the center WL
            imLevel1I = level1[i,WL,0:YInv,0:XInv,0]
            imLevel1Q = level1[i,WL,0:YInv,0:XInv,1]
            imLevel1U = level1[i,WL,0:YInv,0:XInv,2]
            imLevel1V = level1[i,WL,0:YInv,0:XInv,3]

            imLevel2S = level2[i,0:YInv,0:XInv,0]
            imLevel2I = level2[i,0:YInv,0:XInv,1]
            imLevel2A = level2[i,0:YInv,0:XInv,2]

            showInversion(fname, imLevel1I, imLevel1Q, imLevel1U, imLevel1V, imLevel2S, imLevel2I, imLevel2A, imPredictS, imPredictI, imPredictA)


    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()

