import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.optimizers import Adam

# Set up tf.data from test.tfr file
# Load model and weights
# for each batch using sess.run
#   predict
#   plot Y(3) and Yhat(3)
#   add results to kde histogram and statistics
# plot kde histograms
# print overall statistics (MSE)

Version='v1-2'
XDim=64
YDim=64
ZDim=4
WDim=112

XStokes=64
YStokes=64
ZStokes=4
WStokes=112

XMagfld=64
YMagfld=64
ZMagfld=3

sizeBatch=32
nEpochs=50
nExamples=18000
nValid=1800

pathTrain = '/data/hinode/tfr/trn-patch.tfr'  # The TFRecord file containing the training set
pathValid = '/data/hinode/tfr/val-patch.tfr'    # The TFRecord file containing the validation set
pathTest = '/data/hinode/tfr/tst-patch.tfr'    # The TFRecord file containing the test set
pathWeight = './data/patch-%s.h5'%(Version)  # The HDF5 weight file generated for the trained model
pathModel = './data/patch-%s.nn'%(Version)  # The model saved as a JSON file
pathLog = '../logs/patch-%s'%(Version)  # The training log

batchSize=32
nTest=1800
batchN=nTest/batchSize

doNormalize=True
normMean = [11856.75185, 0.339544616, 0.031913142, -1.145931805]
normStdev = [3602.323144, 42.30705892, 40.60409966, 43.49488492]

nm = tf.constant(normMean)
ns = tf.constant(normStdev)

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
    # normalize each Stokes Parameter
    if doNormalize:
        x = tf.reshape(x, (WStokes * YStokes * XStokes, ZStokes))
        x = tf.slice(x, (0, 0), (WStokes * YDim * XDim, ZStokes))
        x = x - nm
        x = x / ns

    # unroll into a 4D array
    x = tf.reshape(x, (WStokes, YStokes, XStokes, ZStokes))
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

    # load the model and the weights
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
    
    # global plot settings
    Nr = 2
    Nc = 2
    
    
    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print('file,WL,I 95PCTL,10PCTL,Min,Max,Mean,Std,',end='')
    print('Q 95PCTL,10PCTL,Min,Max,Mean,Std,',end='')
    print('U 95PCTL,10PCTL,Min,Max,Mean,Std,',end='')
    print('V 95PCTL,10PCTL,Min,Max,Mean,Std')
    # Now we read batches of images and labels and plot them
    n = 0
    for batch_index in range(batchN):
        level1, level2, fname = sess.run([X, Y, Name])
        predict = model.predict(X)
        for i in range(X.shape[0]):
            n += 1
            #imI = normalize(level1[i,:,0:64,0:64,:])
            imI = level1[i,:,0:64,0:64,:]
            print('%f,'%(imI[56,32,32,0]), end='')
            print('%f,'%(imI[56,32,32,1]), end='')
            print('%f,'%(imI[56,32,32,2]), end='')
            print('%f,'%(imI[56,32,32,3]), end='')
            print('%s,'%(fname[i]))
            #break
            #for j in range(X.shape[1]):
            for j in range(0, 1):
                #print(fname[i],end='')
                imI = level1[i,j,0:64,0:64,0]
                #nz = np.nonzero(imI)
                #print('%d,'%(j), end='')
                #print('%f,'%(np.percentile(imI, 95.0)), end='')
                #print('%f,'%(np.percentile(imI, 10.0)), end='')
                #print('%f,'%(np.min(imI)), end='')
                #print('%f,'%(np.max(imI)), end='')
                #print('%f,'%(np.mean(imI)), end='')
                #print('%f,'%(np.std(imI)), end='')
                plt.gray()
                plt.imshow(imI)
                plt.title(fname[i])
                plt.show()
                imQ = level1[i,j,0:512,0:864,1]
                nz = np.nonzero(imQ)
                #plt.gray()
                #plt.imshow(imQ)
                #plt.title(fname[i])
                #plt.show()
                #print('%f,'%(np.percentile(imQ[nz], 95.0)), end='')
                #print('%f,'%(np.percentile(imQ[nz], 10.0)), end='')
                #print('%f,'%(np.min(imQ[nz])), end='')
                #print('%f,'%(np.max(imQ[nz])), end='')
                #print('%f,'%(np.mean(imQ[nz])), end='')
                #print('%f,'%(np.std(imQ[nz])), end='')
                imU = level1[i,j,0:512,0:864,2]
                nz = np.nonzero(imU)
                #plt.gray()
                #plt.imshow(imU)
                #plt.title(fname[i])
                #plt.show()
                #print('%f,'%(np.percentile(imU[nz], 95.0)), end='')
                #print('%f,'%(np.percentile(imU[nz], 10.0)), end='')
                #print('%f,'%(np.min(imU[nz])), end='')
                #print('%f,'%(np.max(imU[nz])), end='')
                #print('%f,'%(np.mean(imU[nz])), end='')
                #print('%f,'%(np.std(imU[nz])), end='')
                imV = level1[i,j,0:512,0:864,3]
                nz = np.nonzero(imV)
                #plt.gray()
                #plt.imshow(imV)
                #plt.title(fname[i])
                #plt.show()
                #print('%f,'%(np.percentile(imV[nz], 95.0)), end='')
                #print('%f,'%(np.percentile(imV[nz], 10.0)), end='')
                #print('%f,'%(np.min(imV[nz])), end='')
                #print('%f,'%(np.max(imV[nz])), end='')
                #print('%f,'%(np.mean(imV[nz])), end='')
                #print('%f'%(np.std(imV[nz])))

            #if keep:
            imLevel2 = level2[i, 0:64,0:64,0]
            plt.imshow(imLevel2)
            plt.show()

            #imLevel2 = level2[i, 0:512,0:864,1]
            #plt.imshow(imLevel2)
            #plt.show()

            #imLevel2 = level2[i, 0:512,0:864,2]
            #plt.imshow(imLevel2)
            #plt.show()


    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()

