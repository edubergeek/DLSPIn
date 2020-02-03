import tensorflow as tf
import time
import numpy as np
from scipy import stats
from scipy import signal
from sklearn import metrics
from dipy.core.geometry import sphere2cart
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape
#from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D, BatchNormalization, Activation
#from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.optimizers import Adam

Model='patch-3d'
Version='5-3'
Checkpoint='187'
Basename="%s-v%s"%(Model,Version)
doNormalize=True
doWLConvolution=True
doPlot=False
useLearningRate=1e-3
sizeBatch=32
#nTest=2545
nTest=1152
nStep=int(nTest/sizeBatch)
#nStep=2
nEpochs=1
useBquv=0

units = (("kG", "deg", "deg"), ("kG", "kG", "kG"))
channel = (("Intensity", "Inclination", "Azimuth"), ("Bq", "Bu", "Bv"))
#symbol = (("m_Int", "m_Inc", "m_Az"), ("$(B_x ^2-B_y ^2)^\frac{1}{2}$", "$(B_xB_y)^\frac{1}{2}$", "$B_z$"))
symbol = (("m_Int", "m_Inc", "m_Az"), ("Bq", "Bu", "Bv"))

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


pathTest = './tfr/val-patch.tfr'    # The TFRecord file containing the test set
pathWeight = './model/%se%s.h5'%(Basename,Checkpoint)  # The HDF5 weight file generated for the trained model
pathModel = './model/%s.nn'%(Basename)  # The model saved as a JSON file

normMean = [11856.75185, 0.339544616, 0.031913142, -1.145931805]
normStdev = [3602.323144, 42.30705892, 40.60409966, 43.49488492]

nm = tf.constant(normMean)
ns = tf.constant(normStdev)

yY = ("Yt-Yp", "Yt-Yp", "Yt-Yp")
yX = ("Yt", "Yt", "Yt")

def kde_pred(y, y_pred, title, yaxis, xaxis):
  y = y.flatten()
  #y = (y - y.min()) / (y.max() - y.min())
  #y = y - (y.max() / 2.0)
  y_pred = y_pred.flatten()
  #y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
  #y_pred = y_pred - (y_pred.max() / 2.0)

  #m1 = y+y_pred
  m1 = y
  m2 = y_pred

  print('RMS err:  %f'%(metrics.mean_squared_error(y, y_pred)))

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
  #aspect_ratio = (y_max - y_min) / (y_pred_max - y_pred_min) 
  #ax.set_aspect(aspect_ratio)
  plt.title(title)
  plt.ylabel(yaxis)
  plt.xlabel(xaxis)
  plt.show()

def kde_error(y, y_pred, title, yaxis, xaxis, unit):
  y = y.flatten()
  #y = (y - y.min()) / (y.max() - y.min())
  #y = y - (y.max() / 2.0)
  y_pred = y_pred.flatten()
  #y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
  #y_pred = y_pred - (y_pred.max() / 2.0)

  #m1 = y+y_pred
  m1 = y
  #m2 = y-y_pred
  m2 = y_pred

  #print('%f,'%(np.percentile(m2, 95.0)), end='')
  #print('%f,'%(np.percentile(imI, 10.0)), end='')
  #print('Min err:  %f'%(np.min(m2)))
  #print('Max err:  %f'%(np.max(m2)))
  #print('Mean err: %f'%(np.mean(m2)))
  #print('Std err:  %f'%(np.std(m2)))
  mad = metrics.mean_absolute_error(y, y_pred)
  mse = metrics.mean_squared_error(y, y_pred)
  print('MAD:      %f'%(mad))
  print('MSE:      %f'%(mse))
  print('Min:      %f'%(np.min(y_pred)))
  print('Max:      %f'%(np.max(y_pred)))

  y_min = m1.min()
  y_max = m1.max()
  y_pred_min = m2.min()
  y_pred_max = m2.max()
  y1 = min(y_min,y_pred_min)
  y2 = max(y_max,y_pred_max)
  X, Y = np.mgrid[y_min:y_max:100j, y_pred_min:y_pred_max:100j]
  #(100, 100)
  positions = np.vstack([X.ravel(), Y.ravel()])
  #(2, 10000)
  values = np.vstack([m1, m2])
  #(128, 64)
  kernel = stats.gaussian_kde(values)
  Z = np.reshape(kernel(positions).T, X.shape)

  #fig, ax = plt.subplots()
  fig, (ax1, ax2) = plt.subplots(1, 2)
  #fig1 = plt.figure(1)
  fig.suptitle(title)
  ax1.set_title("KDE plot")
  ax1.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[y1, y2, y1, y2])
  #ax1.plot(m1, m2, 'k.', markersize=2)
  if np.max(y) > 180.0:
    ax1.set_xlim([0, 500])
    ax1.set_ylim([0, 500])
  else:
    ax1.set_xlim([0, 180])
    ax1.set_ylim([0, 180])
  #aspect_ratio = (y_max - y_min) / (y_pred_max - y_pred_min) 
  #ax1.set_aspect(aspect_ratio)
  ax1.set_ylabel(yaxis)
  ax1.set_xlabel(xaxis)

  ax2.set_title("KDE pdf")
  #hist = ax2.hist(m2, bins=100, normed=True)
  n, b, _ = ax2.hist(m2, bins=100, normed=True)
  bins = n
  #mu = np.mean(m2)
  #sigma = np.std(m2)
  #y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
  #ax2.plot(bins, y, '--')
  ax2.set_ylabel("pdf")
  ax2.set_xlabel(xaxis)
  #ax1.set_xlim([-1, 1])
  #ax1.set_ylim([y_pred_min, y_pred_max])
  aspect_ratio = (m2.max() - m2.min()) / bins.max()
  ax2.set_aspect(aspect_ratio)
  madstr="MAD=%0.2f [%s]"%(mad, unit)
  def get_axis_limits(ax, scale=.9):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    #return xlim[0]*(1-scale), xlim[1]*scale, ylim[0]*(1-scale), ylim[1]*scale
    return xlim[0]*scale, xlim[1]*scale, ylim[0]*scale, ylim[1]*scale

  x1, x2, y1, y2=get_axis_limits(ax2)
  ax2.text(x1, y2, madstr, size=8)
  plt.tight_layout()
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

  #print(model.summary())

  print('-'*30)
  print('Testing the model...')
  print('-'*30)

  sess = K.get_session()
  n = 0

  scores = model.evaluate(dsTest, steps=nStep+1, verbose=1)
  for m in range(0, len(scores)):
    print('%s=%f'%(model.metrics_names[m], scores[m]))

  if doPlot:
    plt.gray()
  r = np.zeros((ZMagfld, (nStep+1)*sizeBatch))
  mae = np.zeros((ZMagfld, (nStep+1)*sizeBatch))
  mse = np.zeros((ZMagfld, (nStep+1)*sizeBatch))

  # add an iterator on the test set to the graph
  diTest = dsTest.make_one_shot_iterator().get_next()
  start = time.time()

  # for each batch in the test set
  for t in range(0, nStep+1):
  #for t in range(0, 1):
    # get the next batch of test examples
    examples = sess.run(diTest)
    x = examples[0]
    print(x.shape)
    y = examples[1]
    print(y.shape)
    #print(y)
    if False:
      plt.imshow(x[1,56,:,:,0])
      plt.show()
      plt.imshow(y[1,:,:,0])
      plt.show()

      plt.imshow(x[1,56,:,:,1])
      plt.show()
      plt.imshow(y[1,:,:,1])
      plt.show()

      plt.imshow(x[1,56,:,:,2])
      plt.show()
      plt.imshow(y[1,:,:,2])
      plt.show()

      plt.imshow(x[1,56,:,:,3])
      plt.show()

    y_pred = model.predict(x, steps=1)
    print(y_pred.shape)
    #print(y_pred)
    if False:
      plt.imshow(y_pred[1,:,:,0])
      plt.show()
      plt.imshow(y_pred[1,:,:,1])
      plt.show()
      plt.imshow(y_pred[1,:,:,2])
      plt.show()

    # convert Intensity units from Gauss to kG
    #y[:,:,:,0] = np.true_divide(y[:,:,:,0], 1000.0)
    #y_pred[:,:,:,0] = np.true_divide(y_pred[:,:,:,0], 1000.0)

    if useBquv:
      # transform from spherical coords to Ramos style Bq,Bu,Bv Cartesian coords
      y[:,:,:,1:] = np.radians(y[:,:,:,1:])
      Bx, By, Bz = sphere2cart(y[:,:,:,0], y[:,:,:,1], y[:,:,:,2])
      Bxsq = np.square(Bx)
      Bysq = np.square(By)
      Bxy = np.multiply(Bx, By)
      Bq = np.sign(Bxsq - Bysq) * np.sqrt(np.abs(Bxsq - Bysq))
      Bu = np.sign(Bxy) * np.sqrt(np.abs(Bxy))
      Bv = Bz
      y[:,:,:,0], y[:,:,:,1], y[:,:,:,2] = Bq, Bu, Bv
      
      y_pred[:,:,:,1:] = np.radians(y_pred[:,:,:,1:])
      Bx, By, Bz = sphere2cart(y_pred[:,:,:,0], y_pred[:,:,:,1], y_pred[:,:,:,2])
      Bxsq = np.square(Bx)
      Bysq = np.square(By)
      Bxy = np.multiply(Bx, By)
      Bq = np.sign(Bxsq - Bysq) * np.sqrt(np.abs(Bxsq - Bysq))
      Bu = np.sign(Bxy) * np.sqrt(np.abs(Bxy))
      Bv = Bz
      y_pred[:,:,:,0], y_pred[:,:,:,1], y_pred[:,:,:,2] = Bq, Bu, Bv


    # for each test example in this batch
    for i in range(x.shape[0]):
    #for i in range(28, 29):
      n += 1
      print('.', end='', flush=True)
  
      # for each output dimension
      for d in range(0, ZMagfld):
        yt = y[i, 0:64, 0:64, d]
        yp = y_pred[i, 0:64, 0:64, d]

        # sum the Pearson Rs to take an average per dimension after the main loop
        pr, pp = stats.pearsonr(yt.flatten(), yp.flatten())
        r[d, n-1] = pr

        mae[d, n-1] = metrics.mean_absolute_error(yt, yp)
        mse[d, n-1] = metrics.mean_squared_error(yt, yp)

        yUnits = units[useBquv]
        yChannel = channel[useBquv]
        ySym = symbol[useBquv]
        if doPlot:
          #xt = x[i, 52, 0:64, 0:64, 0]
          #plt.imshow(xt)
          #plt.show()

          plt.imshow(yt)
          plt.show()
          plt.imshow(yp)
          plt.show()

          if False:
            title = '%s: R=%.6f'%(yChannel[d], pr)
            plt.title(title)
            plt.imshow(corr(yt, yp))
            plt.show()

          if True:
            title = '%s KDE'%(yChannel[d])
            print('%s [%s]'%(yChannel[d], yUnits[d]))
            yleg = "%s [%s]"%(ySym[d], yUnits[d])
            xleg = "%s [%s]"%(ySym[d], yUnits[d])
            kde_error(yt, yp, title, yleg, xleg, yUnits[d])
            print("")

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
  #print(r)
  #print('%d test examples in %d seconds: %.3f'%(n,elapsed, elapsed/n))
  print('\n%d test examples'%(n))
  print('%d seconds elapsed'%(elapsed))
  print('Average inversion time %f.2 seconds'%(elapsed/n))

  rmean = np.mean(r, axis=0)
  rstd = np.std(r, axis=0)
  for m in range(0, ZMagfld):
    print('%s: 2D correlation R mean  %f'%(yChannel[m], rmean[m]))
    print('%s: 2D correlation R stdev %f'%(yChannel[m], rstd[m]))

  mmean = np.mean(mae, axis=0)
  mrms = np.mean(mse, axis=0)
  mstd = np.std(mae, axis=0)
  for m in range(0, ZMagfld):
    print('%s: MAE mean  %f'%(yChannel[m], mmean[m]))
    print('%s: MAE stdev %f'%(yChannel[m], mstd[m]))
    print('%s: MSE mean  %f'%(yChannel[m], mrms[m]))

  if doPlot:
    plt.plot( r[0], marker='o', color='blue', linewidth=2, label="intensity")
    plt.plot( r[1], marker='*', color='red', linewidth=2, label="inclination")
    plt.plot( r[2], marker='x', color='green', linewidth=2, label="azimuth")
    plt.legend()
    plt.show()
  K.clear_session()

print(tf.__version__)
test()
