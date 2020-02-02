import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, Dropout
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras import backend as K

Version='3d-v3-5'
doNormalize=True
doWLConvolution=True
doBatchNorm=False
useCheckpoint=""
useDropout=0.05
baseChannels=12
filterSize=3
#useLearningRate=5e-4
useLearningRate=1e-3
useBatchSize=40
nEpochs=150
startEpoch=0

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

nExamples=19269
#nExamples=19240
nValid=2460
#nValid=2440
#nTest=2545

pathTrain = './tfr/trn-patch.tfr'  # The TFRecord file containing the training set
pathValid = './tfr/val-patch.tfr'    # The TFRecord file containing the validation set
pathTest = './tfr/tst-patch.tfr'    # The TFRecord file containing the test set
pathWeight = './model/patch-%s.h5'%(Version)  # The HDF5 weight file generated for the trained model
pathModel = './model/patch-%s.nn'%(Version)  # The model saved as a JSON file
pathLog = './log/patch-%s'%(Version)  # The training log

#normMean = [12317.92913, 0.332441335, -0.297060628, -0.451666942]
#normStdev = [1397.468631, 65.06104869, 65.06167056, 179.3232721]

normMean = [11856.75185, 0.339544616, 0.031913142, -1.145931805]
normStdev = [3602.323144, 42.30705892, 40.60409966, 43.49488492]
# experiment with scaled down intensity by factor of 10x
#normStdev = [36023.23144, 42.30705892, 40.60409966, 43.49488492]
nm = tf.constant(normMean)
ns = tf.constant(normStdev)

def UNetFromCheckpoint(checkpoint):
  # load the pretrained model weights and network
  # The HDF5 weight file generated for the pretrained model
  pathCheckpointWeights = './model/patch-%s.h5'%(checkpoint)
  print(pathCheckpointWeights)
  pathCheckpointModel = './model/patch-%s.nn'%(checkpoint)
  print(pathCheckpointModel)
  model_file = open(pathCheckpointModel, 'r')
  model_serial = model_file.read()
  model_file.close()
  # load from YAML and create model
  model = model_from_yaml(model_serial)
  # load learned weights into trained model
  model.load_weights(pathCheckpointWeights)
   
  # evaluate loaded model on test data
  #model.compile(optimizer=Adam(lr=1e-5), loss='mse', metrics=['mse'])
  model.compile(optimizer=Adam(lr=useLearningRate), loss='mae', metrics=['mae', 'mse'])
  
  return model



def UNet():
  if len(useCheckpoint) > 0:
    return UNetFromCheckpoint(useCheckpoint)

  features=int(1)
  # image size/receptive field: 64x64/3x3
  if doWLConvolution:
    inputs = Input((WStokes, YDim, XDim, ZStokes))
    if doBatchNorm:
      conv1 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(inputs)
      conv1 = BatchNormalization()(conv1)
      conv1 = Activation("relu")(conv1)
      conv1 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv1)
      conv1 = BatchNormalization()(conv1)
      conv1 = Activation("relu")(conv1)
    else:
      conv1 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), activation='relu', padding='same')(inputs)
      conv1 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='relu', padding='same')(conv1)
      if useDropout > 0:
        conv1 = Dropout(useDropout)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
  else:
    inputs = Input((YDim, XDim, WStokes*ZStokes))
    if doBatchNorm:
      conv1 = Conv2D(int(2*features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(inputs)
      conv1 = BatchNormalization()(conv1)
      conv1 = Activation("relu")(conv1)
      conv1 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv1)
      conv1 = BatchNormalization()(conv1)
      conv1 = Activation("relu")(conv1)
    else:
      conv1 = Conv2D(int(2*features*baseChannels), (filterSize, filterSize), activation='relu', padding='same')(inputs)
      conv1 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='relu', padding='same')(conv1)
      if useDropout > 0:
        conv1 = Dropout(useDropout)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  #conv1 = Conv3D(64, (WDim, 1, 1), use_bias=False, padding='same')(inputs)
  #conv1 = BatchNormalization()(conv1)
  #conv1 = Activation("relu")(conv1)
  #conv1 = Conv3D(64, (WDim, 1, 1), use_bias=False, padding='same')(conv1)
  #conv1 = BatchNormalization()(conv1)
  #conv1 = Activation("relu")(conv1)
  #conv1 = Conv3D(32, (WDim, 1, 1), activation='relu', padding='same')(inputs)
  #conv1 = concatenate([inputs], axis=0)
  #conv1 = Conv3D(32, (1, 3, 3), activation='relu', padding='same')(inputs)
  #conv1 = MaxPooling3D(pool_size=(WDim, 1, 1))(conv1)
  #conv1 = Reshape((YDim, XDim, 32*WDim))(conv1)



  features=features*2
  # image size/receptive field: 32x32/9x9
  if doWLConvolution:
    if doBatchNorm:
      conv2 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(pool1)
      conv2 = BatchNormalization()(conv2)
      conv2 = Activation("relu")(conv2)
      conv2 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv2)
      conv2 = BatchNormalization()(conv2)
      conv2 = Activation("relu")(conv2)
    else:
      conv2 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), activation='relu', padding='same')(pool1)
      conv2 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='relu', padding='same')(conv2)
      if useDropout > 0:
        conv2 = Dropout(useDropout)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
  else:
    if doBatchNorm:
      conv2 = Conv2D(int(2*features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(pool1)
      conv2 = BatchNormalization()(conv2)
      conv2 = Activation("relu")(conv2)
      conv2 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv2)
      conv2 = BatchNormalization()(conv2)
      conv2 = Activation("relu")(conv2)
    else:
      conv2 = Conv2D(int(2*features*baseChannels), (filterSize, filterSize), activation='relu', padding='same')(pool1)
      conv2 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='relu', padding='same')(conv2)
      if useDropout > 0:
        conv2 = Dropout(useDropout)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  features=features*2
  # image size/receptive field: 16x16/27x27
  if doWLConvolution:
    if doBatchNorm:
      conv3 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(pool2)
      conv3 = BatchNormalization()(conv3)
      conv3 = Activation("relu")(conv3)
      conv3 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv3)
      conv3 = BatchNormalization()(conv3)
      conv3 = Activation("relu")(conv3)
    else:
      conv3 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), activation='relu', padding='same')(pool2)
      conv3 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='relu', padding='same')(conv3)
      if useDropout > 0:
        conv3 = Dropout(useDropout)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
  else:
    if doBatchNorm:
      conv3 = Conv2D(int(2*features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(pool2)
      conv3 = BatchNormalization()(conv3)
      conv3 = Activation("relu")(conv3)
      conv3 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv3)
      conv3 = BatchNormalization()(conv3)
      conv3 = Activation("relu")(conv3)
    else:
      conv3 = Conv2D(int(2*features*baseChannels), (filterSize, filterSize), activation='relu', padding='same')(pool2)
      conv3 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='relu', padding='same')(conv3)
      if useDropout > 0:
        conv3 = Dropout(useDropout)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  #conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
  #conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
  #pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

  #conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool4)
  #conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
  #pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

  #conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool5)
  #conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
  #conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
  #conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)

  features=features*2
  # image size/receptive field: 8x8/81x81
  if doWLConvolution:
    if doBatchNorm:
      conv4 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(pool3)
      conv4 = BatchNormalization()(conv4)
      conv4 = Activation("relu")(conv4)
      conv4 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv4)
      conv4 = BatchNormalization()(conv4)
      conv4 = Activation("relu")(conv4)
    else:
      conv4 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), activation='relu', padding='same')(pool3)
      conv4 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='relu', padding='same')(conv4)
      if useDropout > 0:
        conv4 = Dropout(useDropout)(conv4)
  else:
    if doBatchNorm:
      conv4 = Conv2D(int(2*features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(pool3)
      conv4 = BatchNormalization()(conv4)
      conv4 = Activation("relu")(conv4)
      conv4 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv4)
      conv4 = BatchNormalization()(conv4)
      conv4 = Activation("relu")(conv4)
    else:
      conv4 = Conv2D(int(2*features*baseChannels), (filterSize, filterSize), activation='relu', padding='same')(pool3)
      conv4 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='relu', padding='same')(conv4)
      if useDropout > 0:
        conv4 = Dropout(useDropout)(conv4)

  if doWLConvolution:
    up5 = concatenate([Conv3DTranspose(int(features*baseChannels), (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
    if doBatchNorm:
      conv5 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(up5)
      conv5 = BatchNormalization()(conv5)
      conv5 = Activation("relu")(conv5)
      conv5 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv5)
      conv5 = BatchNormalization()(conv5)
      conv5 = Activation("relu")(conv5)
    else:
      conv5 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='relu', padding='same')(up5)
      conv5 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='relu', padding='same')(conv5)
      if useDropout > 0:
        conv5 = Dropout(useDropout)(conv5)
  else:
    up5 = concatenate([Conv2DTranspose(int(features*baseChannels), (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
    if doBatchNorm:
      conv5 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(up5)
      conv5 = BatchNormalization()(conv5)
      conv5 = Activation("relu")(conv5)
      conv5 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv5)
      conv5 = BatchNormalization()(conv5)
      conv5 = Activation("relu")(conv5)
    else:
      conv5 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='relu', padding='same')(up5)
      conv5 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='relu', padding='same')(conv5)
      if useDropout > 0:
        conv5 = Dropout(useDropout)(conv5)

  #up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv4], axis=3)
  #conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
  #conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

  #up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv3], axis=3)
  #conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
  #conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

  #up10 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv9), conv2], axis=3)
  #conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(up10)
  #conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv10)
  features = features / 2
  if doWLConvolution:
    up6 = concatenate([Conv3DTranspose(int(features*baseChannels), (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv2], axis=4)
    if doBatchNorm:
      conv6 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(up6)
      conv6 = BatchNormalization()(conv6)
      conv6 = Activation("relu")(conv6)
      conv6 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv6)
      conv6 = BatchNormalization()(conv6)
      conv6 = Activation("relu")(conv6)
    else:
      conv6 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='relu', padding='same')(up6)
      conv6 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='relu', padding='same')(conv6)
      if useDropout > 0:
        conv6 = Dropout(useDropout)(conv6)
  else:
    up6 = concatenate([Conv2DTranspose(int(features*baseChannels), (2, 2), strides=(2, 2), padding='same')(conv5), conv2], axis=3)
    if doBatchNorm:
      conv6 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(up6)
      conv6 = BatchNormalization()(conv6)
      conv6 = Activation("relu")(conv6)
      conv6 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv6)
      conv6 = BatchNormalization()(conv6)
      conv6 = Activation("relu")(conv6)
    else:
      conv6 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='relu', padding='same')(up6)
      conv6 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='relu', padding='same')(conv6)
      if useDropout > 0:
        conv6 = Dropout(useDropout)(conv6)

  #up11 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv10), conv1], axis=3)
  #conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(up11)
  #conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv11)
  features = features / 2
  if doWLConvolution:
    up7 = concatenate([Conv3DTranspose(int(features*baseChannels), (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv1], axis=4)
    if doBatchNorm:
      conv7 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(up7)
      conv7 = BatchNormalization()(conv7)
      conv7 = Activation("relu")(conv7)
      conv7 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv7)
      conv7 = BatchNormalization()(conv7)
      conv7 = Activation("relu")(conv7)
    else:
      conv7 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='relu', padding='same')(up7)
      conv7 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='relu', padding='same')(conv7)
      if useDropout > 0:
        conv7 = Dropout(useDropout)(conv7)
  else:
    up7 = concatenate([Conv2DTranspose(int(features*baseChannels), (2, 2), strides=(2, 2), padding='same')(conv6), conv1], axis=3)
    if doBatchNorm:
      conv7 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(up7)
      conv7 = BatchNormalization()(conv7)
      conv7 = Activation("relu")(conv7)
      conv7 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv7)
      conv7 = BatchNormalization()(conv7)
      conv7 = Activation("relu")(conv7)
    else:
      conv7 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='relu', padding='same')(up7)
      conv7 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='relu', padding='same')(conv7)
      if useDropout > 0:
        conv7 = Dropout(useDropout)(conv7)

  if doWLConvolution:
    # finally we project 3D to 2D to make a final prediction
    conv8 = Conv3D(ZMagfld, (WStokes, 1, 1), activation='linear')(conv7)
  else:
    # make a final prediction (already 2D)
    conv8 = Conv2D(ZMagfld, (1, 1), activation='linear')(conv7)
  conv8 = Reshape((YDim, XDim, ZMagfld))(conv8)

  model = Model(inputs=[inputs], outputs=[conv8])

  model.compile(optimizer=Adam(lr=useLearningRate), loss='mae', metrics=['mae', 'mse'])

  print(model.summary())

  return model

def train():
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
      #x = tf.subtract(x , nm)
      #x = tf.divide(x, ns)
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

  #construct a TFRecordDataset
  dsTrain = tf.data.TFRecordDataset(pathTrain).map(_parse_record)
  dsTrain = dsTrain.prefetch(useBatchSize)
  dsTrain = dsTrain.shuffle(2*useBatchSize)
  dsTrain = dsTrain.repeat()
  dsTrain = dsTrain.batch(useBatchSize)

  dsValid = tf.data.TFRecordDataset(pathValid).map(_parse_record)
  #dsValid = dsValid.shuffle()
  dsValid = dsValid.prefetch(useBatchSize)
  dsValid = dsValid.repeat()
  dsValid = dsValid.batch(useBatchSize)

  print('-'*30)
  print('Loading model...')
  print('-'*30)
  model = UNet()
  callbacks = [
      tf.keras.callbacks.ModelCheckpoint(pathWeight, verbose=1, monitor='val_loss', save_best_only=True),
      tf.keras.callbacks.TensorBoard(log_dir = pathLog, histogram_freq = 5, write_graph = False, write_images = True)
  ]

  print('-'*30)
  print('Serializing model...')
  print('-'*30)

  # serialize model to JSON
  model_serial = model.to_json()
  with open(pathModel, "w") as yaml_file:
    yaml_file.write(model_serial)

  print('-'*30)
  print('Training model...')
  print('-'*30)

  #print(dsTrain)
  history = model.fit(dsTrain, validation_data=dsValid, validation_steps=int(np.ceil(nValid/useBatchSize)), steps_per_epoch=int(np.ceil(nExamples/useBatchSize)), initial_epoch=startEpoch, epochs=nEpochs, verbose=1, callbacks=callbacks)

  #Y = model.predict(dsTest, steps=1)
  #image = Y[0,:,:,0]
  
  #fig = plt.figure(num='Level 2 - Predicted')

  #plt.gray()
  #plt.imshow(image)
  #plt.show()
  #plt.close(fig)

print(tf.__version__)
train()
