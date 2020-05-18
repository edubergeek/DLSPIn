import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

pathTrain = './data/trn-patch.tfr'  # The TFRecord file containing the training set
pathValid = './data/val-patch.tfr'    # The TFRecord file containing the validation set
pathTest = './data/tst-patch.tfr'    # The TFRecord file containing the test set

batchSize=32
batchN=100

#normMean = [12317.92913, 0.332441335, -0.297060628, -0.451666942]
#normStdev = [1397.468631, 65.06104869, 65.06167056, 179.3232721]
normMean = [11856.75185, 0.339544616, 0.031913142, -1.145931805]
normStdev = [3602.323144, 42.30705892, 40.60409966, 43.49488492]


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
    # normalize each Stokes Parameter
    x = (x - normMean) / normStdev

    
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

    colormap = plt.cm.inferno
    # Now we read batches of images and labels and plot them
    n = 0
    for batch_index in range(batchN):
        level1, level2, fname = sess.run([X, Y, Name])
        for i in range(X.shape[0]):
            n += 1
            imI = level1[i,0,0:64,0:64,0]
            keep=True
            for r in range(0, 64):
                if np.min(imI[r,:]) == np.max(imI[r,:]) or np.sum(imI[r,:]) == 0:
                    keep = False
                    break
            if not keep:
                continue
            keep=True
            for c in range(0, 64):
                if np.min(imI[:,c]) == np.max(imI[:,c]) or np.sum(imI[:,c]) == 0:
                    keep = False
                    break
            if not keep:
                continue
            #for j in range(X.shape[1]):
            for j in range(56, 57):
                keep=False
                print(fname[i],end='')
                imI = level1[i,j,0:64,0:64,0]
                #nz = np.nonzero(imI)
                if np.std(imI) > 1.0:
                    keep=True
                    print('file,WL,I 95PCTL,10PCTL,Min,Max,Mean,Std,',end='')
                    print('Q 95PCTL,10PCTL,Min,Max,Mean,Std,',end='')
                    print('U 95PCTL,10PCTL,Min,Max,Mean,Std,',end='')
                    print('V 95PCTL,10PCTL,Min,Max,Mean,Std')
                    print('%d,'%(j), end='')
                    print('%f,'%(np.percentile(imI, 95.0)), end='')
                    print('%f,'%(np.percentile(imI, 10.0)), end='')
                    print('%f,'%(np.min(imI)), end='')
                    print('%f,'%(np.max(imI)), end='')
                    print('%f,'%(np.mean(imI)), end='')
                    print('%f,'%(np.std(imI)), end='')
                    plt.gray()
                    plt.imshow(imI, cmap=colormap)
                    plt.title(fname[i])
                    plt.show()
                imQ = level1[i,j,0:64,0:64,1]
                nz = np.nonzero(imQ)
                if keep:
                    plt.gray()
                    plt.imshow(imQ, cmap=colormap)
                    plt.title(fname[i])
                    plt.show()
                print('%f,'%(np.percentile(imQ[nz], 95.0)), end='')
                print('%f,'%(np.percentile(imQ[nz], 10.0)), end='')
                print('%f,'%(np.min(imQ[nz])), end='')
                print('%f,'%(np.max(imQ[nz])), end='')
                print('%f,'%(np.mean(imQ[nz])), end='')
                print('%f,'%(np.std(imQ[nz])), end='')
                imU = level1[i,j,0:64,0:64,2]
                nz = np.nonzero(imU)
                if keep:
                    plt.gray()
                    plt.imshow(imU, cmap=colormap)
                    plt.title(fname[i])
                    plt.show()
                print('%f,'%(np.percentile(imU[nz], 95.0)), end='')
                print('%f,'%(np.percentile(imU[nz], 10.0)), end='')
                print('%f,'%(np.min(imU[nz])), end='')
                print('%f,'%(np.max(imU[nz])), end='')
                print('%f,'%(np.mean(imU[nz])), end='')
                print('%f,'%(np.std(imU[nz])), end='')
                imV = level1[i,j,0:64,0:64,3]
                nz = np.nonzero(imV)
                if keep:
                    plt.gray()
                    plt.imshow(imV, cmap=colormap)
                    plt.title(fname[i])
                    plt.show()
                print('%f,'%(np.percentile(imV[nz], 95.0)), end='')
                print('%f,'%(np.percentile(imV[nz], 10.0)), end='')
                print('%f,'%(np.min(imV[nz])), end='')
                print('%f,'%(np.max(imV[nz])), end='')
                print('%f,'%(np.mean(imV[nz])), end='')
                print('%f'%(np.std(imV[nz])))

            if keep:
                imLevel2 = level2[i, 0:64,0:64,0]
                plt.imshow(imLevel2, cmap=colormap)
                plt.show()

                imLevel2 = level2[i, 0:64,0:64,1]
                plt.imshow(imLevel2, cmap=colormap)
                plt.show()

                imLevel2 = level2[i, 0:64,0:64,2]
                plt.imshow(imLevel2, cmap=colormap)
                plt.show()


    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()

