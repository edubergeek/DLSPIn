    #if imageData.shape[1] == YDim:
import os
import numpy as np
import matplotlib.pyplot as plt
import fs
from fs import open_fs
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.optimizers import Adam

dirsep = '/'
csvdelim = ','
pathData='/d/hinode/data'
pathWeight = '../data/test1.h5'  # The HDF5 weight file generated for the trained model
pathModel = '../data/test1.nn'  # The model saved as a JSON file
imageText = "image"
inputText = "*.fits"
outputText = "out"
trainCSV = "./spin.csv"
xdim=32
ydim=32
XDim=875
YDim=512
ZDim=4

wlOffset=10
spNum=0
SPName = ['I', 'Q', 'U', 'V' ]
magNum=0
MagName = ['Strength', 'Inclination', 'Azimuth']

def chunkstring(string, length):
  return (string[0+i:length+i] for i in range(0, len(string), length))

def normalize(img, threshold):
  val = np.percentile(img,threshold)
  img = img / val
  return img


def load_fits(filnam):
  from astropy.io import fits

  hdulist = fits.open(filnam)
  meta = {}
  #gen = chunkstring(hdulist[0].header, 80)
  #for keyval in gen:
  #  for x in keyval.astype('U').split('\n'):
  #    meta = x
  #    print(meta)
  #  #meta.update( dict(x.split('=') for x in np.array_str(keyval, 80).split('\n')) )
  h = list(chunkstring(hdulist[0].header, 80))
  for index, item in enumerate(h):
    m = str(item)
    mh = list(chunkstring(m, 80))
    #print(mh)
    for ix, im in enumerate(mh):
      #print(index, ix, im)
      mm = im.split('/')[0].split('=')
      if len(mm) == 2:
        #print(index, ix, mm[0], mm[1])
        meta[mm[0].strip()] = mm[1].strip()
  nAxes = int(meta['NAXIS'])
  if nAxes == 0:
    # should be checking metadata to verify this is a LEVEL1 image
    nAxes = 3
    maxy, maxx = hdulist[1].data.shape
    data = np.empty((maxy, maxx, 3))
    data[:,:,0] = hdulist[1].data
    data[:,:,1] = hdulist[2].data
    data[:,:,2] = hdulist[3].data
  else:
    data = hdulist[0].data
  data = np.nan_to_num(data)
  #img = data.reshape((maxy, maxx, maxz))
  #img = np.rollaxis(data, 1)
  img = data
  if nAxes == 3:
    maxy, maxx, maxz = data.shape
  else:
    maxy, maxx = data.shape
    maxz = 0
  hdulist.close
  return maxy, maxx, maxz, meta, img

# Generator function to walk path and generate 1 SP3D image set at a time
def process_sp3d(pathData):
  prevImageName=''
  level = 0
  fsDetection = open_fs(pathData)
  img=np.empty((YDim,XDim,ZDim))
  for path in fsDetection.walk.files(search='breadth', filter=[inputText]):
    # process each "in" file of detections
    inName=pathData+path
    #print('Inspecting %s'%(inName))
    #open the warp warp diff image using "image" file
    sub=inName.split(dirsep)
    imageName=sub[-2]
    if imageName != prevImageName:
      if prevImageName != '':
        # New image so wrap up the current image
        # Flip image Y axis
        img = np.flip(img, axis=0)
        yield img, fitsName, level, wl
      # Initialize for a new image
      #print('Parsing %s - %s'%(imageName, path))
      prevImageName = imageName
      fitsName=sub[-1]
      img[:,:]=0
    #else:
    #  print('Appending %s to %s'%(path, imageName))
    #imgName=pathData+dirsep+pointing+dirsep+imageText
    #imgName=inName
    #byteArray=bytearray(np.genfromtxt(imgName, 'S'))
    #imageFile=byteArray.decode()
    imageFile=inName
    #print("Opening image file %s"%(imageFile))
    height, width, depth, imageMeta, imageData = load_fits(imageFile)
    # now the pixels are in the array imageData shape height X width X 1
    # read the truth table from the "out" file
    #for k, v in imageMeta.items():
    #  print(k,v)
    if 'INVCODE' in imageMeta:
      # level 2 FITS file
      level = 2
      dimY, dimX, dimZ = imageData.shape
      #dimZ = 0
      # we should have 3 dimensions, the azimuth, altitude and intensity
      wl = (float(imageMeta['LMIN2']) + float(imageMeta['LMAX2'])) / 2.0
      img[0:dimY,0:dimX,0:dimZ] = imageData
    else:
      # level 1 FITS file
      level = 1
      x = int(imageMeta['SLITINDX'])
      wl = float(imageMeta['CRVAL1'])
      dimZ, dimY, dimX = imageData.shape
      # concatenate the next column of data
      v=np.reshape(imageData[0,:,56+wlOffset],(dimY))
      img[0:dimY,x,0] = v
      v=np.reshape(imageData[1,:,56+wlOffset],(dimY))
      img[0:dimY,x,1] = v
      v=np.reshape(imageData[2,:,56+wlOffset],(dimY))
      img[0:dimY,x,2] = v
      v=np.reshape(imageData[3,:,56+wlOffset],(dimY))
      img[0:dimY,x,3] = v

  if prevImageName != '':
    # New image so wrap up the current image
    # Flip image Y axis
    img = np.flip(img, axis=0)
    yield img, fitsName, level, wl
    fsDetection.close()
  
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

# find input files in the target dir "pathData"
for image, name, level, line in process_sp3d(pathData):
  #outFileHandle=open(trainCSV,'wt')
  #outFileHandle.close()
  print(name, level, image.shape)
  plt.gray()
  if level==1:
    image = normalize(image, 95)
    # forward pass through the model to invert the Level 1 SP images and predict the Level 2 image mag fld maps
    batchIn = np.reshape(image[0:512,0:864,0:4], (1, 512, 864, 4))
    batchOut = model.predict(batchIn)
    imPredict = batchOut[0,:,:,magNum]
    imLevel1 = image[0:512,0:864,spNum]

    # prepare to visualize
    #fig = plt.figure(num='%s %.2fA SP=%s MF=%s'%(name,line,SPName[spNum], MagName[magNum]))
    #fig, axs = plt.subplots(Nr, Nc)
    #fig.suptitle('%s %.2fA SP=%s MF=%s'%(name,line,SPName[spNum],MagName[magNum]))

    # Plot the images
    #axs[0, 0].imshow(imLevel1)
    #axs[0, 1].imshow(imLevel2)
    #axs[1, 0].imshow(imLevel1)
    #axs[1, 1].imshow(imPredict)
    plt.imshow(imLevel1)
    plt.show()
    plt.imshow(imPredict)
    plt.show()
    #plt.close(fig)
  else:
    #fig = plt.figure(num='%s %.2fA SP=%s MF=%s'%(name,line,SPName[spNum], MagName[magNum]))
    imLevel2 = image[0:512,0:864,magNum]
    plt.imshow(imLevel2)
    plt.show()
