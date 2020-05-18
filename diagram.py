import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import fs
from fs import open_fs
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

dirsep = '/'
csvdelim = ','
#pathData='/d/hinode/spin/data/20120306_221040'
#pathData='/d/hinode/spin/data/20140202_041505'
#pathData='/d/hinode/spin/data/20130315_183005'
#pathData='/d/hinode/spin/data/20140704_131342'
#pathWeight = './model/patch-3d-v3-5e65.h5'  # The HDF5 weight file generated for the trained model
#pathModel = './model/patch-3d-v3-5.nn'  # The model saved as a JSON file
#pathWeight = './model/patch-3d-v1-1e146.h5'  # The HDF5 weight file generated for the trained model
#pathModel = './model/patch-3d-v1-1.nn'  # The model saved as a JSON file
Model='patch-3d'
Version='5-1'
Checkpoint='18'
Basename="%s-v%s"%(Model,Version)
doWLConvolution=True
    
imageText = "image"
inputText = "*.fits"
outputText = "out"
trainCSV = "./spin.csv"

pathData='/d/hinode/spin/invert'
pathWeight = './model/%se%s.h5'%(Basename,Checkpoint)  # The HDF5 weight file generated for the trained model
pathModel = './model/%s.nn'%(Basename)  # The model saved as a JSON file

xdim=32
ydim=32
XDim=875
YDim=512
ZDim=4
ZMag=3

WDim=112
WStart=0
WStep=1

XInv=864
YInv=512
ZInv=4
WInv=112

# number of 64x64 patches in a canonical input image
PXDim=64
PYDim=64
PX=13
PY=8

plotWL=56
plotPatch=200

spNum=3
SPName = ['I', 'Q', 'U', 'V' ]
magNum=1
MagName = ['Strength', 'Inclination', 'Azimuth']
MagUnit = ['Gauss', 'deg', 'deg']

normMean = [11856.75185, 0.339544616, 0.031913142, -1.145931805]
normStdev = [3602.323144, 42.30705892, 40.60409966, 43.49488492]

def chunkstring(string, length):
  return (string[0+i:length+i] for i in range(0, len(string), length))

def normalize(img):
  for d in range(0, 4):
    img[:,:,:,d] = img[:,:,:,d] - normMean[d]
    img[:,:,:,d] = img[:,:,:,d] / normStdev[d]
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
    if len(hdulist[1].data.shape) < 2:
      data = np.empty((1, 1, 3))
    else:
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
def process_sp3d(basePath):
  prevImageName=''
  level = 0
  fsDetection = open_fs(basePath)
  img=np.empty((WDim,YDim,XDim,ZDim))
  WInd = list(range(WStart,WStart+WDim*WStep,WStep))
  for path in fsDetection.walk.files(search='breadth', filter=[inputText]):
    # process each "in" file of detections
    inName=basePath+path
    #print('Inspecting %s'%(inName))
    #open the warp warp diff image using "image" file
    sub=inName.split(dirsep)
    imageName=sub[-2]
    if imageName != prevImageName:
      if prevImageName != '':
        # New image so wrap up the current image
        # Flip image Y axis
        #img = np.flip(img, axis=1)
        yield img, fitsName, level, wl, meta
      # Initialize for a new image
      #print('Parsing %s - %s'%(imageName, path))
      prevImageName = imageName
      fitsName=sub[-1]
      # reset image to zeros
      img[:,:,:,:]=0
    #else:
    #  print('Appending %s to %s'%(path, imageName))
    #imgName=basePath+dirsep+pointing+dirsep+imageText
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
      # crop to maximum height
      dimY = min(dimY, YDim)
      # crop to maximum width
      dimX = min(dimX, XDim)
      dimW = 0
      #dimZ = 0
      # we should have 3 dimensions, the azimuth, altitude and intensity
      wl = (float(imageMeta['LMIN2']) + float(imageMeta['LMAX2'])) / 2.0
      img[0,0:dimY,0:dimX,0:dimZ] = imageData[0:dimY,0:dimX,0:dimZ]
      meta = imageMeta
    else:
      # level 1 FITS file
      level = 1
      x = int(imageMeta['SLITINDX'])
      if x < XDim:
        wl = float(imageMeta['CRVAL1']) + (WStart*float(imageMeta['CDELT1']))
        dimZ, dimY, dimX = imageData.shape
        # crop to maximum height
        dimY = min(dimY, YDim)
        # crop to maximum width
        dimX = min(dimX, XDim)
        dimW = WDim
        # concatenate the next column of data
        # 4, 512, 112
        # 1, 512, 9
        a=np.reshape(imageData[0,:dimY,:],(dimY, dimX))
        a = a[:,WInd]
        img[0:dimW,0:dimY,x,0] = np.transpose(a)
        a=np.reshape(imageData[1,:dimY,:],(dimY, dimX))
        a = a[:,WInd]
        img[0:dimW,0:dimY,x,1] = np.transpose(a)
        a=np.reshape(imageData[2,:dimY,:],(dimY, dimX))
        a = a[:,WInd]
        img[0:dimW,0:dimY,x,2] = np.transpose(a)
        a=np.reshape(imageData[3,:dimY,:],(dimY, dimX))
        a = a[:,WInd]
        img[0:dimW,0:dimY,x,3] = np.transpose(a)
        meta = imageMeta

  if prevImageName != '':
    # New image so wrap up the current image
    # Flip image Y axis
    #img = np.flip(img, axis=1)
    yield img, fitsName, level, wl, meta
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
model.summary(line_length=132)
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
 
for layer in model.layers:
  print(layer.name, layer.input_shape)
  print(layer.get_output_at(0).get_shape().as_list())
