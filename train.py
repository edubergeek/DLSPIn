import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

XDim=864
YDim=512
ZDim=4

XStokes=875
YStokes=512
ZStokes=4

XMagfld=875
YMagfld=512
ZMagfld=3

sizeBatch=8
nEpochs=30

pathTrain = '../data/train.tfr'  # The TFRecord file containing the training set
pathValid = '../data/val.tfr'    # The TFRecord file containing the validation set
pathTest = '../data/test.tfr'    # The TFRecord file containing the test set
pathWeight = '../data/test1.h5'  # The HDF5 weight file generated for the trained model

def UNet():
  inputs = Input((YDim, XDim, ZStokes))
  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

  up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
  conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
  conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

  up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
  conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
  conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

  up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
  conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
  conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

  up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
  conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
  conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

  conv10 = Conv2D(ZMagfld, (1, 1), activation='linear')(conv9)

  model = Model(inputs=[inputs], outputs=[conv10])

  model.compile(optimizer=Adam(lr=1e-5), loss='mse', metrics=['mse'])

  return model


def train():
  K.set_image_data_format('channels_last')  # TF dimension ordering in this code
  featdef = {
    #'magfld': tf.FixedLenSequenceFeature(shape=[YMagfld*XMagfld*ZMagfld], dtype=tf.string, allow_missing=True),
    #'stokes': tf.FixedLenSequenceFeature(shape=[YStokes*XStokes*ZStokes], dtype=tf.string, allow_missing=True)
    'magfld': tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True),
    'stokes': tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True)
    }
        
  def _parse_record(example_proto, clip=False):
    """Parse a single record into x and y images"""
    example = tf.parse_single_example(example_proto, featdef)
    #x = tf.decode_raw(example['stokes'], tf.float32)
    x = example['stokes']
    x = tf.reshape(x, (YStokes, XStokes, ZStokes))
    x = tf.slice(x, (0, 0, 0), (YDim, XDim, ZStokes))
    
    #y = tf.decode_raw(example['magfld'], tf.float32)
    y = example['magfld']
    y = tf.reshape(y, (YMagfld, XMagfld, ZMagfld))
    y = tf.slice(y, (0, 0, 0), (YDim, XDim, ZMagfld))

    return x, y

  #construct a TFRecordDataset
  dsTrain = tf.data.TFRecordDataset(pathTrain).map(_parse_record)
  dsTrain = dsTrain.repeat(30)
  dsTrain = dsTrain.shuffle(10).batch(sizeBatch)

  dsValid = tf.data.TFRecordDataset(pathValid).map(_parse_record)
  dsValid = dsValid.repeat(30)
  dsValid = dsValid.shuffle(10).batch(sizeBatch)

  dsTest = tf.data.TFRecordDataset(pathTest).map(_parse_record)
  dsTest = dsValid.repeat(30)
  dsTest = dsValid.shuffle(10).batch(sizeBatch)

  print('-'*30)
  print('Creating and compiling model...')
  print('-'*30)
  model = UNet()
  callbacks = [
      tf.keras.callbacks.ModelCheckpoint(pathWeight, verbose=1, save_best_only=True),
      tf.keras.callbacks.TensorBoard(log_dir='../logs')
  ]

  print('-'*30)
  print('Fitting model...')
  print('-'*30)

  print(dsTrain)
  history = model.fit(dsTrain, validation_data=dsValid, validation_steps=2, steps_per_epoch=10, epochs=nEpochs, verbose=1, callbacks=callbacks)
  #history = model.fit(dsTrain, validation_data=dsValid, epochs=nEpochs, verbose=1, callbacks=callbacks)

  Y = model.predict(dsTest, steps=1)
  image = Y[0,:,:,0]
  
  fig = plt.figure(num='Level 2 - Predicted')

  plt.gray()
  plt.imshow(image)
  plt.show()
  plt.close(fig)

print(tf.__version__)
train()
