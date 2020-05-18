import tensorflow as tf
import numpy as np


pathTrain = './tfr/trn-patch.tfr'  # The TFRecord file containing the training set
pathValid = './tfr/val-patch.tfr'    # The TFRecord file containing the validation set
pathTest = './tfr/tst-patch.tfr'    # The TFRecord file containing the test set

pathIn = pathTest
pathOut = './tfr/tst-patch-active.tfr'  # The TFRecord file containing the sliced output

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

doNormalize=False
normMean = [11856.75185, 0.339544616, 0.031913142, -1.145931805]
normStdev = [3602.323144, 42.30705892, 40.60409966, 43.49488492]

nm = tf.constant(normMean)
ns = tf.constant(normStdev)

batchSize=1
#batchN=22647
#batchN=38685
batchN=5000
nOut=0

def _floatvector_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


with tf.device('/cpu:0'):
  with tf.Session() as sess:
    feature = {
        'magfld':  tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True),
        'stokes':  tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True),
        'name':    tf.FixedLenFeature(shape=[], dtype=tf.string),
        'xcen':    tf.FixedLenFeature(shape=[], dtype=tf.float32),
        'ycen':    tf.FixedLenFeature(shape=[], dtype=tf.float32),
        'xoff':    tf.FixedLenFeature(shape=[], dtype=tf.float32),
        'yoff':    tf.FixedLenFeature(shape=[], dtype=tf.float32),
        'dateobs': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'active':  tf.FixedLenFeature(shape=[], dtype=tf.string)
    }
        
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([pathIn], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(pathOut)


    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    sample = tf.parse_single_example(serialized_example, features=feature)

    active = sample['active']
    magfld = sample['magfld']
    stokes = sample['stokes']
    name = sample['name']
    xcen = sample['xcen']
    ycen = sample['ycen']
    xoff = sample['xoff']
    yoff = sample['yoff']
    dateobs = sample['dateobs']
    # Any preprocessing here ...
    # normalize each Stokes Parameter
    if doNormalize:
        stokes = tf.reshape(stokes, (WStokes * YStokes * XStokes, ZStokes))
        stokes = tf.slice(stokes, (0, 0), (WStokes * YDim * XDim, ZStokes))
        stokes = stokes - nm
        stokes = stokes / ns

    # unroll into a 4D array
    stokes = tf.reshape(stokes, (WStokes, YStokes, XStokes, ZStokes))
    stokes = tf.slice(stokes, (0, 0, 0, 0), (WStokes, YDim, XDim, ZStokes))

    # unroll into a 3D array
    magfld = tf.reshape(magfld, (YMagfld, XMagfld, ZMagfld))
    # use slice to crop the data
    magfld = tf.slice(magfld, (0, 0, 0), (YDim, XDim, ZMagfld))
    
    
    # Creates batches by randomly shuffling tensors
    Active, Magfld, Stokes, Name, Xcen, Ycen, Xoff, Yoff, Dateobs = \
      tf.train.batch([active, magfld, stokes, name, xcen, ycen, xoff, yoff, dateobs], batch_size=batchSize)
  
    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
  
    # Now we read batches of images and labels and plot them
    #print("nTrain", end =" ")
    for batch_index in range(batchN):
      try:
          active, magfld, stokes, name, xcen, ycen, xoff, yoff, dateobs = \
            sess.run([Active, Magfld, Stokes, Name, Xcen, Ycen, Xoff, Yoff, Dateobs])
      except tf.errors.OutOfRangeError:
        #print("End of dataset")  # "End of dataset"
        #nTrain += fname[0].size
        #if example[0] == b'0':
        #  writer.write(sess.run(serialized_example))
        #  nOut += 1
        print("OutOfRange at %d"%(nOut))
        break
  
        # Stop the threads
        #coord.request_stop()
    
        # Wait for threads to stop
        #coord.join(threads)
        #sess.close()
      #print(active, magfld, stokes, name, xcen, ycen, xoff, yoff, dateobs)
      #    'magfld': _floatvector_feature(magfld.tolist()),
      if active[0] == b'1':
        yp=np.reshape(magfld, (YMagfld*XMagfld*ZMagfld))
        xp=np.reshape(stokes, (WStokes*YStokes*XStokes*ZStokes))
        #print(name)
        feature = {
          'magfld': _floatvector_feature(yp.tolist()),
          'stokes': _floatvector_feature(xp.tolist()),
          'xcen': _float_feature(xcen),
          'ycen': _float_feature(ycen),
          'xoff': _float_feature(xoff),
          'yoff': _float_feature(yoff),
          'name': _bytes_feature(name[0]),
          'dateobs': _bytes_feature(dateobs[0]),
          'active': _bytes_feature(active[0])
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        nOut += 1
        print(".", end=" ")
        if batch_index % 100 == 0:
          print("%d"%(nOut))

      #if example["active"][0] == b'1':
      #  sample = sess.run(serialized_example)
      #  writer.write(sample)
      #  nOut += 1
      #  print(".", end=" ")
      #  if batch_index % 100 == 0:
      #    print("%d"%(nOut))
      #for p in range(example[0].size):
      #  print(example[0][p], end=" ")
      #print()
      #nOut += example[0].size

    print('%d training examples output'%(nOut))
    exit
