import os
import numpy as np
import fs
from fs import open_fs

dirsep = '/'
csvdelim = ','
#basePath='/d/hinode/data/20120703_134836/level1'
#20130114_123005
basePath='/d/hinode/data/20180807_210006/level1'
#basePath='/d/hinode/data'


imageText = "image"
inputText = "*.fits"
outputText = "out"
trainCSV = "./spin.csv"
xdim=32
ydim=32
XDim=875
YDim=512

wlOffset=10
spNum=0
SPName = ['I', 'Q', 'U', 'V' ]
magNum=0
MagName = ['Strength', 'Inclination', 'Azimuth']

def chunkstring(string, length):
  return (string[0+i:length+i] for i in range(0, len(string), length))

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
      mm = im.split('/')[0].split('=')
      if len(mm) == 2:
        meta[mm[0].strip()] = mm[1].strip()
  nAxes = int(meta['NAXIS'])
  if nAxes == 0:
    nAxes = 2
    data = hdulist[3].data
  else:
    data = hdulist[0].data
  data = np.nan_to_num(data)
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
  img=np.empty((YDim,XDim))
  for path in fsDetection.walk.files(filter=[inputText],exclude_dirs=['level2']):
    # process each "in" file of detections
    inName=basePath+path
    #open the warp warp diff image using "image" file
    sub=inName.split(dirsep)
    imageName=sub[-2]
    if imageName != prevImageName:
      if prevImageName != '':
        # New image so wrap up the current image
        yield fitsName, imageMeta
      # Initialize for a new image
      print('Parsing %s - %s'%(imageName, path))
      prevImageName = imageName
      fitsName=sub[-1]
      img[:,:]=0
    imageFile=inName
    height, width, depth, imageMeta, imageData = load_fits(imageFile)
    # now the pixels are in the array imageData shape height X width X 1
    # read the truth table from the "out" file

  if prevImageName != '':
    yield fitsName, imageMeta
  

# main
# find input files in the target dir "basePath"
for name, meta in process_sp3d(basePath):
    print('FITS file %s metadata:'%(name))
    for k, v in meta.items():
      print(k,v)
