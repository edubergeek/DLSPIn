import os
import numpy as np
import matplotlib.pyplot as plt
import fs
from fs import open_fs

dirsep = '/'
csvdelim = ','
#basePath="/d/hinode/spin/hao"
basePath='hao/web/csac.hao.ucar.edu/data/hinode/sot/level1/2012/10/21/SP3D'
imageText = "image"
inputText = "*.fits"
outputText = "out"
trainCSV = "./spin.csv"
xdim=32
ydim=32

def chunkstring(string, length):
  return (string[0+i:length+i] for i in range(0, len(string), length))

def normalize(img, bits):
  vlo=np.amin(img)
  vhi=np.amax(img)
  vrg=vhi-vlo
  img = (img - vlo) / vrg * 2**(bits-1)
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
        meta[mm[0]] = mm[1]
  data = hdulist[0].data
  data = np.nan_to_num(data)
  #print(data.shape)
  #img = data.reshape((maxy, maxx, maxz))
  #img = np.rollaxis(data, 1)
  img = data
  #print(img.shape)
  maxy, maxx, maxz = data.shape
#  fits.close(hdulist)
  return maxy, maxx, maxz, meta, img

# main
# find input files in the target dir "basePath"
plt.gray()
outFileHandle=open(trainCSV,'wt')
fsDetection = open_fs(basePath)
img=np.empty((512,875))
# TBD modify to walk and process 1 SP3D image set at a time
for path in fsDetection.walk.files(filter=[inputText]):
  # process each "in" file of detections
  inName=basePath+path
  #print('Inspecting %s'%(inName))
  #open the warp warp diff image using "image" file
  sub=path.split(dirsep)
  pointing=sub[1]
  #imgName=basePath+dirsep+pointing+dirsep+imageText
  #imgName=inName
  #byteArray=bytearray(np.genfromtxt(imgName, 'S'))
  #imageFile=byteArray.decode()
  imageFile=inName
  print("Opening image file %s"%(imageFile))
  height, width, depth, imageMeta, imageData = load_fits(imageFile)
  # now the pixels are in the array imageData shape height X width X 1
  # read the truth table from the "out" file
  #print(imageMeta)
  x = int(imageMeta['SLITINDX'])
  dimZ, dimY, dimX = imageData.shape
  #print('Z dimension=%d'%(dimZ))
  #print('Y dimension=%d'%(dimY))
  #print('X dimension=%d'%(dimX))
  #for z in range(dimZ):
  #  print(imageData[z,0,0])
  #for z in range(dimZ):
  #  print(imageData[z,0,dimX-1])
  #for z in range(dimZ):
  #  print(imageData[z,dimY-1,0])
  #for z in range(dimZ):
  #  print(imageData[z,dimY-1,dimX-1])
  # concatenate the next column of data
  v=np.reshape(imageData[0,:,56],(512))
  xp = x - 1
  xn = x + 1
  img[:,x] = v
  #if x > 0 and x < 874:
  #  img = np.hstack((img[:,:xp],v,img[:,xn:]))
  #else:
  #  if x == 0:
  #    img = np.vstack((v,img[:,xn:]))
  #  else:
  #    img = np.vstack((img[:,:xp],v))

outFileHandle.close()
imgNorm = normalize(img, 8)
plt.imshow(imgNorm)
plt.show()
