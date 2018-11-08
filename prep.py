import os
import numpy as np
import matplotlib.pyplot as plt
import fs
from fs import open_fs

dirsep = '/'
csvdelim = ','
#basePath="/d/hinode/spin/hao"
#basePath='hao/web/csac.hao.ucar.edu/data/hinode/sot/level1/2012/10/21/SP3D'
#basePath='hao/web/csac.hao.ucar.edu/data/hinode/sot/level1'
#basePath='hao/web/csac.hao.ucar.edu/data/hinode/sot'
basePath='/d/hinode/data'
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

#def normalize(img, bits):
#  vlo=np.amin(img)
#  vhi=np.amax(img)
#  vrg=vhi-vlo
#  img = (img - vlo) / vrg * 2**(bits-1)
#  return img

def normalize(img, threshold):
  val = np.percentile(img,threshold)
  img = img / val
  return img

def postage_stamp(img,x,y,w,h):
  iy,ix = img.shape
  xo = int(w/2)
  yo = int(h/2)
  cx = min(max(np.floor(x).astype(int)-xo,0),ix-w)
  cy = min(max(np.floor(iy-y).astype(int)-yo,0),iy-h)
  crop=img[cy:cy+h, cx:cx+w]
  return crop, cx, cy

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
    nAxes = 2
    data = hdulist[3].data
  else:
    data = hdulist[0].data
  data = np.nan_to_num(data)
  #print(data.shape)
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
  img=np.empty((YDim,XDim))
  for path in fsDetection.walk.files(filter=[inputText]):
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
        img = np.flip(img, axis=0)
        yield img, fitsName, level, wl
      # Initialize for a new image
      print('Parsing %s - %s'%(imageName, path))
      prevImageName = imageName
      fitsName=sub[-1]
      img[:,:]=0
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
      dimY, dimX = imageData.shape
      dimZ = 0
      wl = (float(imageMeta['LMIN2']) + float(imageMeta['LMAX2'])) / 2.0
      img[0:dimY,0:dimX] = imageData
    else:
      # level 1 FITS file
      level = 1
      x = int(imageMeta['SLITINDX'])
      wl = float(imageMeta['CRVAL1'])
      dimZ, dimY, dimX = imageData.shape
      # concatenate the next column of data
    #if imageData.shape[1] == YDim:
      v=np.reshape(imageData[spNum,:,56+wlOffset],(dimY))
      img[0:dimY,x] = v

  if prevImageName != '':
    # New image so wrap up the current image
    # Flip image Y axis
    img = np.flip(img, axis=0)
    yield img, fitsName, level, wl
  

# main
# find input files in the target dir "basePath"
for image, name, level, line in process_sp3d(basePath):
  #outFileHandle=open(trainCSV,'wt')
  #outFileHandle.close()
  image = normalize(image, 95)
  if level==1:
    fig = plt.figure(num='%s Level %d %.2fA SP=%s'%(name,level,line,SPName[spNum]))
  else:
    fig = plt.figure(num='%s Level %d %.2fA %s'%(name,level,line,MagName[magNum]))

  plt.gray()
  plt.imshow(image)
  plt.show()
  plt.close(fig)
