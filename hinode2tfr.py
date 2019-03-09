from random import shuffle
import glob
import os
import sys
import numpy as np
import fs
from fs import open_fs
import tensorflow as tf

dirsep = '/'
csvdelim = ','
basePath='/d/hinode/data'
imageText = "image"
inputText = "*.fits"
outputText = "out"
trainCSV = "./spin.csv"
xdim=32
ydim=32
XDim=875
YDim=512
ZDim=4

WDim=9
WStart=0
WStep=5

pTest = 0.1
pVal = 0.1
nCopies=1

train_filename = '../data/train.tfr'  # the TFRecord file containing the training set
val_filename = '../data/val.tfr'      # the TFRecord file containing the validation set
test_filename = '../data/test.tfr'    # the TFRecord file containing the test set

def chunkstring(string, length):
  return (string[0+i:length+i] for i in range(0, len(string), length))

# filter images on mean level 1 intensity pixel value and position on sun
# True means don't use the image and False means OK to use the image
def isFiltered(img, nz, meta):
  if np.std(img[nz]) >= 0.07:
    yc = float(meta['YCEN'])
    if yc >= -300.0 and yc <= 300.0:
      xc = float(meta['XCEN'])
      if xc >= -700.0 and xc <= 700.0:
        return False
  return True

def normalize(img, nz, threshold):
  val = np.percentile(img[nz],threshold)
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
        img = np.flip(img, axis=1)
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
    img = np.flip(img, axis=1)
    yield img, fitsName, level, wl, meta
    fsDetection.close()
  
def _floatvector_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

np.random.seed()

# open the TFRecords file
train_writer = tf.python_io.TFRecordWriter(train_filename)

# open the TFRecords file
val_writer = tf.python_io.TFRecordWriter(val_filename)

# open the TFRecords file
test_writer = tf.python_io.TFRecordWriter(test_filename)

# find input files in the target dir "basePath"
# it is critical that pairs are produced reliably first level2 then level1
# for each level2 (Y) file
i = nExamples = nTrain = nVal = nTest = 0

img=np.empty((WDim,YDim,XDim,ZDim))
#nz=np.empty((ZDim))

for image, name, level, line, meta in process_sp3d(basePath):
  if level == 2:
    # image is the level2 magnetic field prediction per pixel (Y)
    ya=np.reshape(image[0,0:YDim,0:XDim,0:3],(YDim*XDim*3))
    nExamples += 1
    print(nExamples, name, level, line)
  else:
    # image is the level1 Stokes params for several WLs of the line of interest (X)
    # filter image using first wavelength (axis 0) and SP=I (axis 3)
    nz0 = image[:,:,:,0].nonzero()
    nz1 = image[:,:,:,1].nonzero()
    nz2 = image[:,:,:,2].nonzero()
    nz3 = image[:,:,:,3].nonzero()
    img[:,:,:,0] = normalize(image[:,:,:,0], nz0, 95.0)
    img[:,:,:,1] = normalize(image[:,:,:,1], nz1, 95.0)
    img[:,:,:,2] = normalize(image[:,:,:,2], nz2, 95.0)
    img[:,:,:,3] = normalize(image[:,:,:,3], nz3, 95.0)
    img_filt = image[0,:,:,0]
    nz_filt = img_filt.nonzero()
    if isFiltered(img_filt, nz_filt, meta):
      print(name, level, line, 'Filtered')
      nExamples -= 1
    else:
      print(name, level, line)
      xa=np.reshape(img,(WDim*YDim*XDim*ZDim))
      feature = {'magfld': _floatvector_feature(ya.tolist()), 'stokes': _floatvector_feature(xa.tolist()), 'name': _bytes_feature(name.encode('utf-8')) }
      # Create an example protocol buffer
      example = tf.train.Example(features=tf.train.Features(feature=feature))
  
      # roll the dice to see if this is a train, val or test example
      # and write it to the appropriate TFRecordWriter
      for clone in range(nCopies):
        i += 1
        roll = np.random.random()
        if roll >= (pVal + pTest):
          # Serialize to string and write on the file
          train_writer.write(example.SerializeToString())
          nTrain += 1
        elif roll >= pTest:
          # Serialize to string and write on the file
          val_writer.write(example.SerializeToString())
          nVal += 1
        else:
          # Serialize to string and write on the file
          test_writer.write(example.SerializeToString())
          nTest += 1

  if not nExamples % 100 and i > 0:
    print('%d examples: %03.1f%% train, %03.1f%%validate, %03.1f%%test.'%(nExamples, 100.0*nTrain/i, 100.0*nVal/i, 100.0*nTest/i))
    sys.stdout.flush()
  
  #if nExamples == 1000:
  #  break

train_writer.close()
val_writer.close()
test_writer.close()
print('%d examples: %03.1f%% train, %03.1f%%validate, %03.1f%%test.'%(nExamples, 100.0*nTrain/nExamples, 100.0*nVal/nExamples, 100.0*nTest/nExamples))
sys.stdout.flush()

