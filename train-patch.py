import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

Version='v1-4'
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
nExamples=10880
nValid=1024

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

doNormalize=True
nm = tf.constant(normMean)
ns = tf.constant(normStdev)

def UNet():
  inputs = Input((WStokes, YDim, XDim, ZStokes))
  #conv1 = Conv3D(64, (WDim, 1, 1), use_bias=False, padding='same')(inputs)
  #conv1 = BatchNormalization()(conv1)
  #conv1 = Activation("relu")(conv1)
  #conv1 = Conv3D(64, (WDim, 1, 1), use_bias=False, padding='same')(conv1)
  #conv1 = BatchNormalization()(conv1)
  #conv1 = Activation("relu")(conv1)
  #conv1 = Conv3D(32, (WDim, 1, 1), activation='relu', padding='same')(inputs)
  conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
  conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv1)
  pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
  #conv1 = Conv3D(16, (WDim, 3, 3), activation='relu', padding='same')(conv1)
  #conv1 = MaxPooling3D(pool_size=(WDim, 1, 1))(conv1)
  #conv1 = Reshape((YDim, XDim, 32))(conv1)
  #conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
  #conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
  #pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool1)
  conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv2)
  pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
  #conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
  #conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
  #pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool2)
  conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv3)
  pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
  #conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
  #conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
  #pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  #conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)
  #conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
  #pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

  #conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool4)
  #conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
  #pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

  #conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool5)
  #conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
  conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
  conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)

  up5 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
  conv5 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up5)
  conv5 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv5)
  #up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv5], axis=3)
  #conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
  #conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

  #up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv4], axis=3)
  #conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
  #conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

  #up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv3], axis=3)
  #conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
  #conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

  #up10 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv9), conv2], axis=3)
  #conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(up10)
  #conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv10)
  up6 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv2], axis=4)
  conv6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up6)
  conv6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv6)

  #up11 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv10), conv1], axis=3)
  #conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(up11)
  #conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv11)
  up7 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv1], axis=4)
  conv7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up7)
  conv7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv7)

  #conv12 = Conv2D(ZMagfld, (1, 1), activation='linear')(conv11)
  conv8 = Conv3D(ZMagfld, (WStokes, 1, 1), activation='linear')(conv7)
  #conv8 = MaxPooling3D(pool_size=(WStokes, 1, 1))(conv7)
  conv8 = Reshape((YDim, XDim, ZMagfld))(conv8)

  model = Model(inputs=[inputs], outputs=[conv8])

  model.compile(optimizer=Adam(lr=1e-5), loss='mse', metrics=['mse'])

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
    
    y = example['magfld']
    y = tf.reshape(y, (YMagfld, XMagfld, ZMagfld))
    y = tf.slice(y, (0, 0, 0), (YDim, XDim, ZMagfld))

#    name = example['name']

#    return x, y, name
    return x, y

  #construct a TFRecordDataset
  dsTrain = tf.data.TFRecordDataset(pathTrain).map(_parse_record)
  dsTrain = dsTrain.prefetch(sizeBatch)
  dsTrain = dsTrain.shuffle(2*sizeBatch)
  dsTrain = dsTrain.repeat()
  dsTrain = dsTrain.batch(sizeBatch)

  dsValid = tf.data.TFRecordDataset(pathValid).map(_parse_record)
  #dsValid = dsValid.shuffle()
  dsValid = dsValid.prefetch(sizeBatch)
  dsValid = dsValid.repeat()
  dsValid = dsValid.batch(sizeBatch)

  print('-'*30)
  print('Loading model...')
  print('-'*30)
  model = UNet()
  callbacks = [
      tf.keras.callbacks.ModelCheckpoint(pathWeight, verbose=1, monitor='val_loss', save_best_only=True),
      tf.keras.callbacks.TensorBoard(log_dir=pathLog)
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
  history = model.fit(dsTrain, validation_data=dsValid, validation_steps=int(np.ceil(nValid/sizeBatch)), steps_per_epoch=int(np.ceil(nExamples/sizeBatch)), epochs=nEpochs, verbose=1, callbacks=callbacks)

  #Y = model.predict(dsTest, steps=1)
  #image = Y[0,:,:,0]
  
  #fig = plt.figure(num='Level 2 - Predicted')

  #plt.gray()
  #plt.imshow(image)
  #plt.show()
  #plt.close(fig)

print(tf.__version__)
train()
