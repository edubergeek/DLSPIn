import tensorflow as tf
import time
import numpy as np
from scipy import stats
from scipy import signal
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape
#from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D, BatchNormalization, Activation
#from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.optimizers import Adam

Version='v4-5e90'
doNormalize=True
doWLConvolution=True
doPlot=False
useLearningRate=1e-3

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
nEpochs=1
nTest=1024
nStep=int(nTest/sizeBatch)

pathTrain = '/data/hinode/tfr/trn-patch.tfr'  # The TFRecord file containing the training set
pathValid = '/data/hinode/tfr/val-patch.tfr'    # The TFRecord file containing the validation set
pathTest = '/data/hinode/tfr/tst-patch.tfr'    # The TFRecord file containing the test set
pathWeight = './data/patch-%s.h5'%(Version)  # The HDF5 weight file generated for the trained model
pathModel = './data/patch-%s.nn'%(Version)  # The model saved as a JSON file
pathLog = '../logs/patch-%s'%(Version)  # The training log

#normMean = [12317.92913, 0.332441335, -0.297060628, -0.451666942]
#normStdev = [1397.468631, 65.06104869, 65.06167056, 179.3232721]

normMean = [11856.75185, 0.339544616, 0.031913142, -1.145931805]
normStdev = [3602.323144, 42.30705892, 40.60409966, 43.49488492]

nm = tf.constant(normMean)
ns = tf.constant(normStdev)

yChannel = ("Intensity", "Inclination", "Azimuth")
yUnits = ("Gauss", "Degrees", "Degrees")
yY = ("Yt-Yp", "Yt-Yp", "Yt-Yp")
yX = ("Yt", "Yt", "Yt")

def kde(y, y_pred, title, yaxis, xaxis):
  y = y.flatten()
  #y = (y - y.min()) / (y.max() - y.min())
  #y = y - (y.max() / 2.0)
  y_pred = y_pred.flatten()
  #y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
  #y_pred = y_pred - (y_pred.max() / 2.0)

  #m1 = y+y_pred
  m1 = y
  m2 = y-y_pred
  y_min = m1.min()
  y_max = m1.max()
  y_pred_min = m2.min()
  y_pred_max = m2.max()
  X, Y = np.mgrid[y_min:y_max:100j, y_pred_min:y_pred_max:100j]
  #(100, 100)
  positions = np.vstack([X.ravel(), Y.ravel()])
  #(2, 10000)
  values = np.vstack([m1, m2])
  #(128, 64)
  kernel = stats.gaussian_kde(values)
  Z = np.reshape(kernel(positions).T, X.shape)

  fig, ax = plt.subplots()
  ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[y_min, y_max, y_pred_min, y_pred_max])
  ax.plot(m1, m2, 'k.', markersize=2)
  ax.set_xlim([y_min, y_max])
  ax.set_ylim([y_pred_min, y_pred_max])
  plt.title(title)
  plt.ylabel(yaxis)
  plt.xlabel(xaxis)
  plt.show()

def corr(y, y_pred):
  corr = signal.correlate2d(y, y_pred)
  return corr

def UNet():
  # open the trained model network
  model_file = open(pathModel, 'r')
  model_serial = model_file.read()
  model_file.close()
  # load from YAML and create model
  model = model_from_yaml(model_serial)
  # load learned weights into trained model
  model.load_weights(pathWeight)
   
  # evaluate loaded model on test data
  #model.compile(optimizer=Adam(lr=1e-5), loss='mse', metrics=['mse'])
  model.compile(optimizer=Adam(lr=useLearningRate), loss='mae', metrics=['mae', 'mse'])
  
  return model


def test():
  K.set_image_data_format('channels_last')  # TF dimension ordering in this code
  featdef = {
    #'magfld': tf.FixedLenSequenceFeature(shape=[YMagfld*XMagfld*ZMagfld], dtype=tf.string, allow_missing=True),
    #'stokes': tf.FixedLenSequenceFeature(shape=[YStokes*XStokes*ZStokes], dtype=tf.string, allow_missing=True)
    'magfld': tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True),
    'stokes': tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True),
#    'name':   tf.FixedLenFeature(shape=[], dtype=tf.string)
    }
       
  def _parse_record(example_proto, clip=False):
    """Parse a single record into x and y images"""
    example = tf.parse_single_example(example_proto, featdef)
    #x = tf.decode_raw(example['stokes'], tf.float32)
    x = example['stokes']

    # normalize Stokes Parameters
    if doNormalize:
      x = tf.reshape(x, (WStokes * YStokes * XStokes, ZStokes))
      x = tf.slice(x, (0, 0), (WStokes * YDim * XDim, ZStokes))
      x = x - nm
      x = x / ns

    x = tf.reshape(x, (WStokes, YStokes, XStokes, ZStokes))
    x = tf.slice(x, (0, 0, 0, 0), (WStokes, YDim, XDim, ZStokes))

    if doWLConvolution == False:
      # unstack wavelength dimension (axis=0)
      # now we have a list of WDim tensors of shape (YDim*XDim, ZStokes)
      wl = tf.unstack(x, axis=0)
      # concatenate the wavelengths in the channel (Stokes Param) dimension (axis=2)
      x = tf.concat(wl, axis=2)
      x = tf.slice(x, (0, 0, 0), (YDim, XDim, WStokes*ZStokes))
    
    y = example['magfld']
    y = tf.reshape(y, (YMagfld, XMagfld, ZMagfld))
    y = tf.slice(y, (0, 0, 0), (YDim, XDim, ZMagfld))

#    name = example['name']

#    return x, y, name
    return x, y

  print('-'*30)
  print('Opening test dataset ...')
  print('-'*30)
  dsTest = tf.data.TFRecordDataset(pathTest).map(_parse_record)
  dsTest = dsTest.prefetch(sizeBatch)
  dsTest = dsTest.repeat(1)
  dsTest = dsTest.batch(sizeBatch)

  print('-'*30)
  print('Creating and compiling model...')
  print('-'*30)
  model = UNet()

  print(model.summary())

  print('-'*30)
  print('Testing the model...')
  print('-'*30)

  sess = K.get_session()
  n = 0

  scores = model.evaluate(dsTest, steps=nStep+1, verbose=1)

  if doPlot:
    plt.gray()
  r = np.zeros((ZMagfld))

  # add an iterator on the test set to the graph
  diTest = dsTest.make_one_shot_iterator().get_next()
  start = time.time()

  # for each batch in the test set
  for t in range(0, nStep+1):
    # get the next batch of test examples
    examples = sess.run(diTest)
    x = examples[0]
    #print(x.shape)
    y = examples[1]
    #print(y.shape)
    y_pred = model.predict(x, steps=1)
    #print(y_pred.shape)

    #print(y_pred)

    #print(model.layers[0].input_shape)
    #y = model.layers[0].input
    #print(y)
    #y_in = model.get_layer("Y",0)
    #level1, level2, fname = sess.run([X, Y, Name])
    #level1, level2 = sess.run([X, Y])

    # for each test example in this batch
    for i in range(x.shape[0]):
      n += 1
      print('.', end='', flush=True)
  
      # for each output dimension
      for d in range(0, 3):
        yt = y[i, 0:64, 0:64, d]
        yp = y_pred[i, 0:64, 0:64, d]

        # sum the Pearson Rs to take an average per dimension after the main loop
        pr, pp = stats.pearsonr(yt.flatten(), yp.flatten())
        r[d] += pr

        if doPlot:
          #xt = x[i, 52, 0:64, 0:64, 0]
          #plt.imshow(xt)
          #plt.show()

          plt.imshow(yt)
          plt.show()
          plt.imshow(yp)
          plt.show()

          title = '%s: R=%.6f'%(yChannel[d], pr)
          plt.title(title)
          plt.imshow(corr(yt, yp))
          plt.show()

          title = '%s KDE (%s)'%(yChannel[d], yUnits[d])
          kde(yt, yp, title, yY[d], yX[d])

      #imI = normalize(level1[i,:,0:64,0:64,:])
      #imI = level1[i,:,0:64,0:64,:]
      #print('%f,'%(imI[56,32,32,0]), end='')
      #print('%f,'%(imI[56,32,32,1]), end='')
      #print('%f,'%(imI[56,32,32,2]), end='')
      #print('%f,'%(imI[56,32,32,3]), end='')
      #print('%s,'%(fname[i]))
      #break
      #for j in range(X.shape[1]):
      #if False: #for j in range(0, 1):
      #  #print(fname[i],end='')
      #  imI = level1[i,j,0:64,0:64,0]
      #  #nz = np.nonzero(imI)
      #  #print('%d,'%(j), end='')
      #  #print('%f,'%(np.percentile(imI, 95.0)), end='')
      #  #print('%f,'%(np.percentile(imI, 10.0)), end='')
      #  #print('%f,'%(np.min(imI)), end='')
      #  #print('%f,'%(np.max(imI)), end='')
      #  #print('%f,'%(np.mean(imI)), end='')
      #  #print('%f,'%(np.std(imI)), end='')
      #  plt.imshow(imI)
      #  #plt.title(fname[i])
      #  plt.show()
      #  imQ = level1[i,j,0:64,0:864,1]
      #  plt.gray()
      #  plt.imshow(imQ)
      #  #plt.title(fname[i])
      #  plt.show()
      #  #print('%f,'%(np.percentile(imQ[nz], 95.0)), end='')
      #  #print('%f,'%(np.percentile(imQ[nz], 10.0)), end='')
      #  #print('%f,'%(np.min(imQ[nz])), end='')
      #  #print('%f,'%(np.max(imQ[nz])), end='')
      #  #print('%f,'%(np.mean(imQ[nz])), end='')
      #  #print('%f,'%(np.std(imQ[nz])), end='')
      #  imU = level1[i,j,0:64,0:864,2]
      #  #nz = np.nonzero(imU)
      #  plt.gray()
      #  plt.imshow(imU)
      #  #plt.title(fname[i])
      #  plt.show()
      #  #print('%f,'%(np.percentile(imU[nz], 95.0)), end='')
      #  #print('%f,'%(np.percentile(imU[nz], 10.0)), end='')
      #  #print('%f,'%(np.min(imU[nz])), end='')
      #  #print('%f,'%(np.max(imU[nz])), end='')
      #  #print('%f,'%(np.mean(imU[nz])), end='')
      #  #print('%f,'%(np.std(imU[nz])), end='')
      #  imV = level1[i,j,0:64,0:864,3]
      #  #nz = np.nonzero(imV)
      #  plt.gray()
      #  plt.imshow(imV)
      #  #plt.title(fname[i])
      #  plt.show()
      #  #print('%f,'%(np.percentile(imV[nz], 95.0)), end='')
      #  #print('%f,'%(np.percentile(imV[nz], 10.0)), end='')
      #  #print('%f,'%(np.min(imV[nz])), end='')
      #  #print('%f,'%(np.max(imV[nz])), end='')
      #  #print('%f,'%(np.mean(imV[nz])), end='')
      #  #print('%f'%(np.std(imV[nz])))


  #scores = model.evaluate(diTest, steps=1, verbose=1)
  #print(scores)
  #scores = model.evaluate(dsTest.take(2), steps=1, verbose=1)
  #y_pred = model.predict(dsTest, steps=1)
  #print(y_pred)

  #print('\nevaluate result: loss={}, mae={}'.format(*scores))
  #image = Y[0,:,:,0]
  
  #fig = plt.figure(num='Level 2 - Predicted')

  #plt.gray()
  #plt.imshow(image)
  #plt.show()
  #plt.close(fig)
  end = time.time()
  elapsed = end - start
  r /= n
  #print(r)
  #print('%d test examples in %d seconds: %.3f'%(n,elapsed, elapsed/n))
  print('\n%d test examples'%(n))
  print('%d seconds elapsed'%(elapsed))
  print('Average inversion time %f.2 seconds'%(elapsed/n))
  print('Average inversion intensity 2D correlation R=%f'%(r[0]))
  print('Average inversion inclination 2D correlation R=%f'%(r[1]))
  print('Average inversion azimuth 2D correlation R=%f'%(r[2]))
  for m in range(0, len(scores)):
    print('%s=%f'%(model.metrics_names[m], scores[m]))
  K.clear_session()

print(tf.__version__)
test()
