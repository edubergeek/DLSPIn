import sherpa
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, Dropout
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras import backend as K

Version='3d-v8-1'
doNormalize=True
doWLConvolution=True
doBatchNorm=False
useCheckpoint=""
useDropout=0.0
useLeakyReLU=0.0
baseChannels=16
filterSize=3
useLearningRate=1e-5
#useLearningRate=1e-6
useBatchSize=16
nEpochs=5
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



def UNet(trial):
  if len(useCheckpoint) > 0:
    return UNetFromCheckpoint(useCheckpoint)

  baseChannels=trial.parameters['units']
  useLearningRate=trial.parameters['lr']
  useDropout=trial.parameters['dropout']
  #doWLConvolution=trial.parameters['3D']
  print(baseChannels)
  print(useLearningRate)
  print(useDropout)
  print(doWLConvolution)

  features=int(1)
  # image size/receptive field: 64x64/3x3
  if doWLConvolution:
    inputs = Input((WStokes, YDim, XDim, ZStokes))
    if doBatchNorm:
      conv1 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(inputs)
      conv1 = BatchNormalization()(conv1)
      if useLeakyReLU > 0:
        conv1 = LeakyReLU(useLeakyReLU)(conv1)
      else:
        conv1 = Activation("relu")(conv1)
      conv1 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv1)
      conv1 = BatchNormalization()(conv1)
      if useLeakyReLU > 0:
        conv1 = LeakyReLU(useLeakyReLU)(conv1)
      else:
        conv1 = Activation("relu")(conv1)
    else:
      conv1 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), padding='same')(inputs)
      #conv1 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), activation='relu', padding='same')(inputs)
      if useLeakyReLU > 0:
        conv1 = LeakyReLU(useLeakyReLU)(conv1)
      else:
        conv1 = Activation("relu")(conv1)
      conv1 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), padding='same')(conv1)
      if useLeakyReLU > 0:
        conv1 = LeakyReLU(useLeakyReLU)(conv1)
      else:
        conv1 = Activation("relu")(conv1)
      if useDropout > 0:
        conv1 = Dropout(useDropout)(conv1)
    #pool1 = conv1
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
  else:
    inputs = Input((YDim, XDim, WStokes*ZStokes))
    if doBatchNorm:
      conv1 = Conv2D(int(2*features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(inputs)
      conv1 = BatchNormalization()(conv1)
      if useLeakyReLU > 0:
        conv1 = LeakyReLU(useLeakyReLU)(conv1)
      else:
        conv1 = Activation("relu")(conv1)
      conv1 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv1)
      conv1 = BatchNormalization()(conv1)
      if useLeakyReLU > 0:
        conv1 = LeakyReLU(useLeakyReLU)(conv1)
      else:
        conv1 = Activation("relu")(conv1)
    else:
      conv1 = Conv2D(int(2*features*baseChannels), (filterSize, filterSize), padding='same')(inputs)
      if useLeakyReLU > 0:
        conv1 = LeakyReLU(useLeakyReLU)(conv1)
      else:
        conv1 = Activation("relu")(conv1)
      conv1 = Conv2D(int(features*baseChannels), (filterSize, filterSize), padding='same')(conv1)
      if useLeakyReLU > 0:
        conv1 = LeakyReLU(useLeakyReLU)(conv1)
      else:
        conv1 = Activation("relu")(conv1)
      if useDropout > 0:
        conv1 = Dropout(useDropout)(conv1)
    #pool1 = conv1
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)


  features=features*2
  # image size/receptive field: 32x32/9x9
  if doWLConvolution:
    if doBatchNorm:
      conv2 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(pool1)
      conv2 = BatchNormalization()(conv2)
      if useLeakyReLU > 0:
        conv2 = LeakyReLU(useLeakyReLU)(conv2)
      else:
        conv2 = Activation("relu")(conv2)
      conv2 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv2)
      conv2 = BatchNormalization()(conv2)
      if useLeakyReLU > 0:
        conv2 = LeakyReLU(useLeakyReLU)(conv2)
      else:
        conv2 = Activation("relu")(conv2)
    else:
      conv2 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), padding='same')(pool1)
      if useLeakyReLU > 0:
        conv2 = LeakyReLU(useLeakyReLU)(conv2)
      else:
        conv2 = Activation("relu")(conv2)
      conv2 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), padding='same')(conv2)
      if useLeakyReLU > 0:
        conv2 = LeakyReLU(useLeakyReLU)(conv2)
      else:
        conv2 = Activation("relu")(conv2)
      if useDropout > 0:
        conv2 = Dropout(useDropout)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
  else:
    if doBatchNorm:
      conv2 = Conv2D(int(2*features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(pool1)
      conv2 = BatchNormalization()(conv2)
      if useLeakyReLU > 0:
        conv2 = LeakyReLU(useLeakyReLU)(conv2)
      else:
        conv2 = Activation("relu")(conv2)
      conv2 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv2)
      conv2 = BatchNormalization()(conv2)
      if useLeakyReLU > 0:
        conv2 = LeakyReLU(useLeakyReLU)(conv2)
      else:
        conv2 = Activation("relu")(conv2)
    else:
      conv2 = Conv2D(int(2*features*baseChannels), (filterSize, filterSize), padding='same')(pool1)
      if useLeakyReLU > 0:
        conv2 = LeakyReLU(useLeakyReLU)(conv2)
      else:
        conv2 = Activation("relu")(conv2)
      conv2 = Conv2D(int(features*baseChannels), (filterSize, filterSize), padding='same')(conv2)
      if useLeakyReLU > 0:
        conv2 = LeakyReLU(useLeakyReLU)(conv2)
      else:
        conv2 = Activation("relu")(conv2)
      if useDropout > 0:
        conv2 = Dropout(useDropout)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  features=features*2
  # image size/receptive field: 16x16/27x27
  if doWLConvolution:
    if doBatchNorm:
      conv3 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(pool2)
      conv3 = BatchNormalization()(conv3)
      if useLeakyReLU > 0:
        conv3 = LeakyReLU(useLeakyReLU)(conv3)
      else:
        conv3 = Activation("relu")(conv3)
      conv3 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv3)
      conv3 = BatchNormalization()(conv3)
      if useLeakyReLU > 0:
        conv3 = LeakyReLU(useLeakyReLU)(conv3)
      else:
        conv3 = Activation("relu")(conv3)
    else:
      conv3 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), padding='same')(pool2)
      if useLeakyReLU > 0:
        conv3 = LeakyReLU(useLeakyReLU)(conv3)
      else:
        conv3 = Activation("relu")(conv3)
      conv3 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), padding='same')(conv3)
      if useLeakyReLU > 0:
        conv3 = LeakyReLU(useLeakyReLU)(conv3)
      else:
        conv3 = Activation("relu")(conv3)
      if useDropout > 0:
        conv3 = Dropout(useDropout)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
  else:
    if doBatchNorm:
      conv3 = Conv2D(int(2*features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(pool2)
      conv3 = BatchNormalization()(conv3)
      if useLeakyReLU > 0:
        conv3 = LeakyReLU(useLeakyReLU)(conv3)
      else:
        conv3 = Activation("relu")(conv3)
      conv3 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv3)
      conv3 = BatchNormalization()(conv3)
      if useLeakyReLU > 0:
        conv3 = LeakyReLU(useLeakyReLU)(conv3)
      else:
        conv3 = Activation("relu")(conv3)
    else:
      conv3 = Conv2D(int(2*features*baseChannels), (filterSize, filterSize), padding='same')(pool2)
      if useLeakyReLU > 0:
        conv3 = LeakyReLU(useLeakyReLU)(conv3)
      else:
        conv3 = Activation("relu")(conv3)
      conv3 = Conv2D(int(features*baseChannels), (filterSize, filterSize), padding='same')(conv3)
      if useLeakyReLU > 0:
        conv3 = LeakyReLU(useLeakyReLU)(conv3)
      else:
        conv3 = Activation("relu")(conv3)
      if useDropout > 0:
        conv3 = Dropout(useDropout)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  features=features*2
  # image size/receptive field: 8x8/81x81
  if doWLConvolution:
    if doBatchNorm:
      conv4 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(pool3)
      conv4 = BatchNormalization()(conv4)
      if useLeakyReLU > 0:
        conv4 = LeakyReLU(useLeakyReLU)(conv4)
      else:
        conv4 = Activation("relu")(conv4)
      conv4 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv4)
      conv4 = BatchNormalization()(conv4)
      if useLeakyReLU > 0:
        conv4 = LeakyReLU(useLeakyReLU)(conv4)
      else:
        conv4 = Activation("relu")(conv4)
    else:
      conv4 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), padding='same')(pool3)
      if useLeakyReLU > 0:
        conv4 = LeakyReLU(useLeakyReLU)(conv4)
      else:
        conv4 = Activation("relu")(conv4)
      conv4 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), padding='same')(conv4)
      if useLeakyReLU > 0:
        conv4 = LeakyReLU(useLeakyReLU)(conv4)
      else:
        conv4 = Activation("relu")(conv4)
      if useDropout > 0:
        conv4 = Dropout(useDropout)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
  else:
    if doBatchNorm:
      conv4 = Conv2D(int(2*features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(pool3)
      conv4 = BatchNormalization()(conv4)
      if useLeakyReLU > 0:
        conv4 = LeakyReLU(useLeakyReLU)(conv4)
      else:
        conv4 = Activation("relu")(conv4)
      conv4 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv4)
      conv4 = BatchNormalization()(conv4)
      if useLeakyReLU > 0:
        conv4 = LeakyReLU(useLeakyReLU)(conv4)
      else:
        conv4 = Activation("relu")(conv4)
    else:
      conv4 = Conv2D(int(2*features*baseChannels), (filterSize, filterSize), padding='same')(pool3)
      if useLeakyReLU > 0:
        conv4 = LeakyReLU(useLeakyReLU)(conv4)
      else:
        conv4 = Activation("relu")(conv4)
      conv4 = Conv2D(int(features*baseChannels), (filterSize, filterSize), padding='same')(conv4)
      if useLeakyReLU > 0:
        conv4 = LeakyReLU(useLeakyReLU)(conv4)
      else:
        conv4 = Activation("relu")(conv4)
      if useDropout > 0:
        conv4 = Dropout(useDropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)


  features=features*2
  # image size/receptive field: 4x4/162x162
  if doWLConvolution:
    if doBatchNorm:
      conv5 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(pool4)
      conv5 = BatchNormalization()(conv5)
      if useLeakyReLU > 0:
        conv5 = LeakyReLU(useLeakyReLU)(conv5)
      else:
        conv5 = Activation("relu")(conv5)
      conv5 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv5)
      conv5 = BatchNormalization()(conv5)
      if useLeakyReLU > 0:
        conv5 = LeakyReLU(useLeakyReLU)(conv5)
      else:
        conv5 = Activation("relu")(conv5)
    else:
      conv5 = Conv3D(int(2*features*baseChannels), (filterSize, filterSize, filterSize), padding='same')(pool4)
      if useLeakyReLU > 0:
        conv5 = LeakyReLU(useLeakyReLU)(conv5)
      else:
        conv5 = Activation("relu")(conv5)
      conv5 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), padding='same')(conv5)
      if useLeakyReLU > 0:
        conv5 = LeakyReLU(useLeakyReLU)(conv5)
      else:
        conv5 = Activation("relu")(conv5)
      if useDropout > 0:
        conv5 = Dropout(useDropout)(conv5)
  else:
    if doBatchNorm:
      conv5 = Conv2D(int(2*features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(pool4)
      conv5 = BatchNormalization()(conv5)
      if useLeakyReLU > 0:
        conv5 = LeakyReLU(useLeakyReLU)(conv5)
      else:
        conv5 = Activation("relu")(conv5)
      conv5 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv5)
      conv5 = BatchNormalization()(conv5)
      if useLeakyReLU > 0:
        conv5 = LeakyReLU(useLeakyReLU)(conv5)
      else:
        conv5 = Activation("relu")(conv5)
    else:
      conv5 = Conv2D(int(2*features*baseChannels), (filterSize, filterSize), padding='same')(pool4)
      if useLeakyReLU > 0:
        conv5 = LeakyReLU(useLeakyReLU)(conv5)
      else:
        conv5 = Activation("relu")(conv5)
      conv5 = Conv2D(int(features*baseChannels), (filterSize, filterSize), padding='same')(conv5)
      if useLeakyReLU > 0:
        conv5 = LeakyReLU(useLeakyReLU)(conv5)
      else:
        conv5 = Activation("relu")(conv5)
      if useDropout > 0:
        conv5 = Dropout(useDropout)(conv5)

  if doWLConvolution:
    up6 = concatenate([Conv3DTranspose(int(features*baseChannels), (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv4], axis=4)
    if doBatchNorm:
      conv6 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(up6)
      conv6 = BatchNormalization()(conv6)
      if useLeakyReLU > 0:
        conv6 = LeakyReLU(useLeakyReLU)(conv6)
      else:
        conv6 = Activation("relu")(conv6)
      conv6 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv6)
      conv6 = BatchNormalization()(conv6)
      if useLeakyReLU > 0:
        conv6 = LeakyReLU(useLeakyReLU)(conv6)
      else:
        conv6 = Activation("relu")(conv6)
    else:
      conv6 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), padding='same')(up6)
      if useLeakyReLU > 0:
        conv6 = LeakyReLU(useLeakyReLU)(conv6)
      else:
        conv6 = Activation("relu")(conv6)
      conv6 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), padding='same')(conv6)
      if useLeakyReLU > 0:
        conv6 = LeakyReLU(useLeakyReLU)(conv6)
      else:
        conv6 = Activation("relu")(conv6)
      if useDropout > 0:
        conv6 = Dropout(useDropout)(conv6)
  else:
    up6 = concatenate([Conv2DTranspose(int(features*baseChannels), (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    if doBatchNorm:
      conv6 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(up6)
      conv6 = BatchNormalization()(conv6)
      if useLeakyReLU > 0:
        conv6 = LeakyReLU(useLeakyReLU)(conv6)
      else:
        conv6 = Activation("relu")(conv6)
      conv6 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv6)
      conv6 = BatchNormalization()(conv6)
      if useLeakyReLU > 0:
        conv6 = LeakyReLU(useLeakyReLU)(conv6)
      else:
        conv6 = Activation("relu")(conv6)
    else:
      conv6 = Conv2D(int(features*baseChannels), (filterSize, filterSize), padding='same')(up6)
      if useLeakyReLU > 0:
        conv6 = LeakyReLU(useLeakyReLU)(conv6)
      else:
        conv6 = Activation("relu")(conv6)
      conv6 = Conv2D(int(features*baseChannels), (filterSize, filterSize), padding='same')(conv6)
      if useLeakyReLU > 0:
        conv6 = LeakyReLU(useLeakyReLU)(conv6)
      else:
        conv6 = Activation("relu")(conv6)
      if useDropout > 0:
        conv6 = Dropout(useDropout)(conv6)

  features = features / 2
  if doWLConvolution:
    up7 = concatenate([Conv3DTranspose(int(features*baseChannels), (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv3], axis=4)
    if doBatchNorm:
      conv7 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(up7)
      conv7 = BatchNormalization()(conv7)
      if useLeakyReLU > 0:
        conv7 = LeakyReLU(useLeakyReLU)(conv7)
      else:
        conv7 = Activation("relu")(conv7)
      conv7 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv7)
      conv7 = BatchNormalization()(conv7)
      if useLeakyReLU > 0:
        conv7 = LeakyReLU(useLeakyReLU)(conv7)
      else:
        conv7 = Activation("relu")(conv7)
    else:
      conv7 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), padding='same')(up7)
      if useLeakyReLU > 0:
        conv7 = LeakyReLU(useLeakyReLU)(conv7)
      else:
        conv7 = Activation("relu")(conv7)
      conv7 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), padding='same')(conv7)
      if useLeakyReLU > 0:
        conv7 = LeakyReLU(useLeakyReLU)(conv7)
      else:
        conv7 = Activation("relu")(conv7)
      if useDropout > 0:
        conv7 = Dropout(useDropout)(conv7)
  else:
    up7 = concatenate([Conv2DTranspose(int(features*baseChannels), (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    if doBatchNorm:
      conv7 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(up7)
      conv7 = BatchNormalization()(conv7)
      if useLeakyReLU > 0:
        conv7 = LeakyReLU(useLeakyReLU)(conv7)
      else:
        conv7 = Activation("relu")(conv7)
      conv7 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv7)
      conv7 = BatchNormalization()(conv7)
      if useLeakyReLU > 0:
        conv7 = LeakyReLU(useLeakyReLU)(conv7)
      else:
        conv7 = Activation("relu")(conv7)
    else:
      conv7 = Conv2D(int(features*baseChannels), (filterSize, filterSize), padding='same')(up7)
      if useLeakyReLU > 0:
        conv7 = LeakyReLU(useLeakyReLU)(conv7)
      else:
        conv7 = Activation("relu")(conv7)
      conv7 = Conv2D(int(features*baseChannels), (filterSize, filterSize), padding='same')(conv7)
      if useLeakyReLU > 0:
        conv7 = LeakyReLU(useLeakyReLU)(conv7)
      else:
        conv7 = Activation("relu")(conv7)
      if useDropout > 0:
        conv7= Dropout(useDropout)(conv7)

  features = features / 2
  if doWLConvolution:
    up8 = concatenate([Conv3DTranspose(int(features*baseChannels), (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv2], axis=4)
    if doBatchNorm:
      conv8 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(up8)
      conv8 = BatchNormalization()(conv8)
      if useLeakyReLU > 0:
        conv8 = LeakyReLU(useLeakyReLU)(conv8)
      else:
        conv8 = Activation("relu")(conv8)
      conv8 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv8)
      conv8 = BatchNormalization()(conv8)
      if useLeakyReLU > 0:
        conv8 = LeakyReLU(useLeakyReLU)(conv8)
      else:
        conv8 = Activation("relu")(conv8)
    else:
      conv8 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), padding='same')(up8)
      if useLeakyReLU > 0:
        conv8 = LeakyReLU(useLeakyReLU)(conv8)
      else:
        conv8 = Activation("relu")(conv8)
      conv8 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), padding='same')(conv8)
      if useLeakyReLU > 0:
        conv8 = LeakyReLU(useLeakyReLU)(conv8)
      else:
        conv8 = Activation("relu")(conv8)
      if useDropout > 0:
        conv8 = Dropout(useDropout)(conv8)
  else:
    up7 = concatenate([Conv2DTranspose(int(features*baseChannels), (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    if doBatchNorm:
      conv8 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(up7)
      conv8 = BatchNormalization()(conv8)
      if useLeakyReLU > 0:
        conv8 = LeakyReLU(useLeakyReLU)(conv8)
      else:
        conv8 = Activation("relu")(conv8)
      conv8 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv8)
      conv8 = BatchNormalization()(conv8)
      if useLeakyReLU > 0:
        conv8 = LeakyReLU(useLeakyReLU)(conv8)
      else:
        conv8 = Activation("relu")(conv8)
    else:
      conv8 = Conv2D(int(features*baseChannels), (filterSize, filterSize), padding='same')(up7)
      if useLeakyReLU > 0:
        conv8 = LeakyReLU(useLeakyReLU)(conv8)
      else:
        conv8 = Activation("relu")(conv8)
      conv8 = Conv2D(int(features*baseChannels), (filterSize, filterSize), padding='same')(conv8)
      if useLeakyReLU > 0:
        conv8 = LeakyReLU(useLeakyReLU)(conv8)
      else:
        conv8 = Activation("relu")(conv8)
      if useDropout > 0:
        conv8 = Dropout(useDropout)(conv8)

  features = features / 2
  if doWLConvolution:
    up9 = concatenate([Conv3DTranspose(int(features*baseChannels), (filterSize, filterSize, filterSize), strides=(2, 2, 2), padding='same')(conv8), conv1], axis=4)
    if doBatchNorm:
      conv9 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(up9)
      conv9 = BatchNormalization()(conv9)
      if useLeakyReLU > 0:
        conv9 = LeakyReLU(useLeakyReLU)(conv9)
      else:
        conv9 = Activation("relu")(conv9)
      conv9 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv9)
      conv9 = BatchNormalization()(conv9)
      if useLeakyReLU > 0:
        conv9 = LeakyReLU(useLeakyReLU)(conv9)
      else:
        conv9 = Activation("relu")(conv9)
    else:
      conv9 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), padding='same')(up9)
      if useLeakyReLU > 0:
        conv9 = LeakyReLU(useLeakyReLU)(conv9)
      else:
        conv9 = Activation("relu")(conv9)
      conv9 = Conv3D(int(features*baseChannels), (filterSize, filterSize, filterSize), padding='same')(conv9)
      if useLeakyReLU > 0:
        conv9 = LeakyReLU(useLeakyReLU)(conv9)
      else:
        conv9 = Activation("relu")(conv9)
      if useDropout > 0:
        conv9 = Dropout(useDropout)(conv9)
  else:
    up9 = concatenate([Conv2DTranspose(int(features*baseChannels), (filterSize, filterSize), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    if doBatchNorm:
      conv9 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(up9)
      conv9 = BatchNormalization()(conv9)
      if useLeakyReLU > 0:
        conv9 = LeakyReLU(useLeakyReLU)(conv9)
      else:
        conv9 = Activation("relu")(conv9)
      conv9 = Conv2D(int(features*baseChannels), (filterSize, filterSize), activation='linear', padding='same', use_bias=False)(conv9)
      conv9 = BatchNormalization()(conv9)
      if useLeakyReLU > 0:
        conv9 = LeakyReLU(useLeakyReLU)(conv9)
      else:
        conv9 = Activation("relu")(conv9)
    else:
      conv9 = Conv2D(int(features*baseChannels), (filterSize, filterSize), padding='same')(up9)
      if useLeakyReLU > 0:
        conv9 = LeakyReLU(useLeakyReLU)(conv9)
      else:
        conv9 = Activation("relu")(conv9)
      conv9 = Conv2D(int(features*baseChannels), (filterSize, filterSize), padding='same')(conv9)
      if useLeakyReLU > 0:
        conv9 = LeakyReLU(useLeakyReLU)(conv9)
      else:
        conv9 = Activation("relu")(conv9)
      if useDropout > 0:
        conv9 = Dropout(useDropout)(conv9)

  if doWLConvolution:
    # finally we project 3D to 2D to make a final prediction
    conv10 = Conv3D(ZMagfld, (WStokes, 1, 1), activation='linear')(conv9)
  else:
    # make a final prediction (already 2D)
    conv10 = Conv2D(ZMagfld, (1, 1), activation='linear')(conv9)
  conv10 = Reshape((YDim, XDim, ZMagfld))(conv10)

  model = Model(inputs=[inputs], outputs=[conv10])

  model.compile(optimizer=Adam(lr=useLearningRate), loss='mae', metrics=['mae', 'mse'])

  print(model.summary())

  return model

def train():
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
  dsValid = dsValid.prefetch(useBatchSize)
  dsValid = dsValid.shuffle(2*useBatchSize)
  dsValid = dsValid.repeat(nEpochs*2)
  dsValid = dsValid.batch(useBatchSize)

  parameters = [sherpa.Choice('units', [2, 4, 8, 16]),
                sherpa.Continuous(name='lr', range=[1e-5, 1e-1], scale='log'),
                sherpa.Continuous(name='dropout', range=[0.0, 0.5], scale='linear')
                ]
#                sherpa.Choice(name='3D', range=[False, True])
  alg = sherpa.algorithms.RandomSearch(max_num_trials=50)
  study = sherpa.Study(parameters=parameters, algorithm=alg, lower_is_better=True, dashboard_port=8880)

  for trial in study:
    K.set_image_data_format('channels_last')  # TF dimension ordering in this code
    model = UNet(trial)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(pathWeight, verbose=1, monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir = pathLog, histogram_freq = 5, write_graph = False, write_images = True),
        study.keras_callback(trial, objective_name='val_loss')
    ]

    # serialize model to JSON
    #model_serial = model.to_json()
    #with open(pathModel, "w") as yaml_file:
    #  yaml_file.write(model_serial)

    # Initialize session - necessary because of Sherpa outer training loop
    K.set_session(tf.Session(graph=model.output.graph))
    init = tf.global_variables_initializer()
    K.get_session().run(init)
    #sess.run(tf.local_variables_initializer())

    history = model.fit(dsTrain, validation_data=dsValid, validation_steps=int(np.ceil(nValid/useBatchSize)), steps_per_epoch=int(np.ceil(nExamples/useBatchSize)), initial_epoch=startEpoch, epochs=nEpochs, verbose=1, callbacks=callbacks)

    study.finalize(trial)
    #K.clear_session()


print(tf.__version__)
train()
