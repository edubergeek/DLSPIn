import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

XDim=864
YDim=512
ZDim=4
WDim=112

XStokes=875
YStokes=512
ZStokes=4
WStokes=112

XMagfld=875
YMagfld=512
ZMagfld=3

pathTrain = './data/trn-x.tfr'  # The TFRecord file containing the training set
pathValid = './data/val-x.tfr'    # The TFRecord file containing the validation set
pathTest = './data/tst-x.tfr'    # The TFRecord file containing the test set

batchSize=1
batchN=75

with tf.Session() as sess:
    feature = {
        'magfld': tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True),
        'stokes': tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True),
        'name':   tf.FixedLenFeature(shape=[], dtype=tf.string)
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

    print('file,WL,I 95PCTL,10PCTL,Min,Max,Mean,Std,',end='')
    print('Q 95PCTL,10PCTL,Min,Max,Mean,Std,',end='')
    print('U 95PCTL,10PCTL,Min,Max,Mean,Std,',end='')
    print('V 95PCTL,10PCTL,Min,Max,Mean,Std')
    # Now we read batches of images and labels and plot them
    for batch_index in range(batchN):
        level1, level2, fname = sess.run([X, Y, Name])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                print(fname[i],end='')
                imI = level1[i,j,0:512,0:864,0]
                nz = np.nonzero(imI)
                #plt.gray()
                #plt.imshow(imI)
                #plt.title(fname[i])
                #plt.show()
                print('%d,'%(j), end='')
                print('%f,'%(np.percentile(imI[nz], 95.0)), end='')
                print('%f,'%(np.percentile(imI[nz], 10.0)), end='')
                print('%f,'%(np.min(imI[nz])), end='')
                print('%f,'%(np.max(imI[nz])), end='')
                print('%f,'%(np.mean(imI[nz])), end='')
                print('%f,'%(np.std(imI[nz])), end='')
                imQ = level1[i,j,0:512,0:864,1]
                nz = np.nonzero(imQ)
                #plt.gray()
                #plt.imshow(imQ)
                #plt.title(fname[i])
                #plt.show()
                print('%f,'%(np.percentile(imQ[nz], 95.0)), end='')
                print('%f,'%(np.percentile(imQ[nz], 10.0)), end='')
                print('%f,'%(np.min(imQ[nz])), end='')
                print('%f,'%(np.max(imQ[nz])), end='')
                print('%f,'%(np.mean(imQ[nz])), end='')
                print('%f,'%(np.std(imQ[nz])), end='')
                imU = level1[i,j,0:512,0:864,2]
                nz = np.nonzero(imU)
                #plt.gray()
                #plt.imshow(imU)
                #plt.title(fname[i])
                #plt.show()
                print('%f,'%(np.percentile(imU[nz], 95.0)), end='')
                print('%f,'%(np.percentile(imU[nz], 10.0)), end='')
                print('%f,'%(np.min(imU[nz])), end='')
                print('%f,'%(np.max(imU[nz])), end='')
                print('%f,'%(np.mean(imU[nz])), end='')
                print('%f,'%(np.std(imU[nz])), end='')
                imV = level1[i,j,0:512,0:864,3]
                nz = np.nonzero(imV)
                #plt.gray()
                #plt.imshow(imV)
                #plt.title(fname[i])
                #plt.show()
                print('%f,'%(np.percentile(imV[nz], 95.0)), end='')
                print('%f,'%(np.percentile(imV[nz], 10.0)), end='')
                print('%f,'%(np.min(imV[nz])), end='')
                print('%f,'%(np.max(imV[nz])), end='')
                print('%f,'%(np.mean(imV[nz])), end='')
                print('%f'%(np.std(imV[nz])))

            #imLevel2 = level2[i, 0:512,0:864,0]
            #plt.imshow(imLevel2)
            #plt.show()


    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()

