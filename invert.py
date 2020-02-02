import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib.cm as cm
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
Version='3-5'
Checkpoint='131'
Basename="%s-v%s"%(Model,Version)
doWLConvolution=True
nPatch=2  #how many overlapping patches are averaged per pixel
# number of 64x64 patches in a canonical input image
PXDim=64
PYDim=64
PX=13
PY=8

    
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

plotWLi=50
plotWLq=72
plotWLu=72
plotWLv=72
plotPatch=2000

spNum=3
SPName = ['I', 'Q', 'U', 'V' ]
magNum=1
MagName = ['Strength', 'Inclination', 'Azimuth']
MagUnit = ['Gauss', 'deg.', 'deg.']

normMean = [11856.75185, 0.339544616, 0.031913142, -1.145931805]
normStdev = [3602.323144, 42.30705892, 40.60409966, 43.49488492]

def blue_red_colormap(low, high):
  # Make a user-defined colormap.
  cm1 = mpc.LinearSegmentedColormap.from_list("MyCmapName",["b","r"])

  # Make a normalizer that will map the min max values
  cnorm = mpc.Normalize(vmin=low,vmax=high)

  # Turn these into an object that can be used to map time values to colors and
  # can be passed to plt.colorbar().
  cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
  cpick.set_array([])
  return cpick

def rounddown(n, decimals=0):
    multiplier = 10 ** decimals
    return np.floor(n * multiplier) / multiplier

def roundup(n, decimals=0):
    multiplier = 10 ** decimals
    return np.ceil(n * multiplier) / multiplier

def chunkstring(string, length):
  return (string[0+i:length+i] for i in range(0, len(string), length))

def normalize(img):
  for d in range(0, 4):
    img[:,:,:,d] = img[:,:,:,d] - normMean[d]
    img[:,:,:,d] = img[:,:,:,d] / normStdev[d]
  return img

def denormalize(img):
  for d in range(0, 4):
    img[:,:,:,d] = img[:,:,:,d] * normStdev[d]
    img[:,:,:,d] = img[:,:,:,d] + normMean[d]
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
 
# evaluate loaded model on test data
model.compile(optimizer=Adam(lr=1e-3), loss='mae', metrics=['mae','mse'])

# global plot settings
Nr = 2
Nc = 2

def invert_patch(SP,MF):
  # 13 * 8 * 2 * 2 = 416
  patchesPerImage = PX * PY * nPatch * nPatch
  # 64 / 2 * (2-1) = 32
  pyoff = int(PYDim/nPatch*(nPatch-1))
  pxoff = int(PXDim/nPatch*(nPatch-1))
  py1 = pyoff
  py2 = YDim-pyoff
  px1 = pxoff
  px2 = XDim-pxoff
  # min(13 * 2, 
  pydim = int(PYDim / nPatch)
  pxdim = int(PXDim / nPatch)
  pymax = YDim-PYDim
  pxmax = XDim-PXDim
  npy = PY * nPatch
  npx = PX * nPatch
  batch = np.zeros((patchesPerImage, WDim, PYDim, PXDim, ZDim))
  # parse an input image into patches
  n = 0
  for yi in range(0, npy):
    for xi in range(0, npx):
      y = yi * pydim
      x = xi * pxdim
      y2 = y+PYDim
      x2 = x+PXDim
      if y2 <= YDim and x2 <= XDim:
        batch[n,:,:,:,:] = SP[:,y:y2,x:x2,:]
      n+=1

  if False:
    plt.imshow(batch[plotPatch,plotWLi,:,:,0])
    plt.show()
    plt.imshow(batch[plotPatch,plotWLq,:,:,1])
    plt.show()
    plt.imshow(batch[plotPatch,plotWLu,:,:,2])
    plt.show()
    plt.imshow(batch[plotPatch,plotWLv,:,:,3])
    plt.show()

  # invert using the SPIN model
    
  if doWLConvolution == False:
    # unstack wavelength dimension (axis=0)
    # now we have a list of WDim tensors of shape (YDim*XDim, ZStokes)
    #wl = tf.unstack(x, axis=0)
    wl = np.copy(np.moveaxis(batch, 1, -2))
    # concatenate the wavelengths in the channel (Stokes Param) dimension (axis=2)
    #x = tf.concat(wl, axis=2)
    batch = np.reshape(wl, (patchesPerImage, PYDim, PXDim, WDim*ZDim))
    
  inversion = model.predict(batch)
  # now reassemble the predicted output image from the inverted patches
  mag = np.zeros((YDim, XDim, ZMag))
  n = 0
  #for yi in range(0, PY):
  #  for xi in range(0, PX):
  #    y = yi * PYDim
  #    x = xi * PXDim
  #    mag[y:y+PYDim,x:x+PXDim,:] = inversion[n,:,:,:]
  for yi in range(0, npy):
    for xi in range(0, npx):
      y = yi * pydim
      x = xi * pxdim
      y2 = y+PYDim
      x2 = x+PXDim
      if y2 <= YInv and x2 <= XInv:
        mag[y:y2,x:x2,:] += inversion[n,:,:,:]

      if n == plotPatch:
        plt.title("Patch %d"%(yi * PX + xi))
        plt.imshow(MF[0,y:y+PYDim,x:x+PXDim,0])
        plt.show()
        plt.imshow(inversion[n,:,:,0])
        plt.show()
        plt.imshow(MF[0,y:y+PYDim,x:x+PXDim,1])
        plt.show()
        plt.imshow(inversion[n,:,:,1])
        plt.show()
        plt.imshow(MF[0,y:y+PYDim,x:x+PXDim,2])
        plt.show()
        plt.imshow(inversion[n,:,:,2])
        plt.show()
      n+=1
  #mag[py1:py2,px1:px2,:] = np.divide(mag[py1:py2,px1:px2,:], nPatch)
  # each corner 32 x 32 * 4
  # each edge 0,32 and XInv-PX * 2
  mag = np.divide(mag, nPatch*nPatch)
  mag[0:pydim,0:pxdim,:] = np.multiply(mag[0:pydim,0:pxdim,:], nPatch*nPatch)
  mag[YInv-pydim:YInv,0:pxdim,:] = np.multiply(mag[YInv-pydim:YInv,0:pxdim,:], nPatch*nPatch)
  mag[0:pydim,XInv-pxdim:XInv,:] = np.multiply(mag[0:pydim,XInv-pxdim:XInv,:], nPatch*nPatch)
  mag[YInv-pydim:YInv,XInv-pxdim:XInv,:] = np.multiply(mag[YInv-pydim:YInv,XInv-pxdim:XInv,:], nPatch*nPatch)

  #mag[pydim:YInv-pydim,XInv-pxdim:XInv,:] = np.multiply(mag[pydim:YInv-pydim,pxdim:XInv-pxdim,:], nPatch)
  mag[0:pydim,pxdim:XInv-pxdim,:] = np.multiply(mag[0:pydim,pxdim:XInv-pxdim,:], nPatch)
  mag[YInv-pydim:YInv,pxdim:XInv-pxdim,:] = np.multiply(mag[YInv-pydim:YInv,pxdim:XInv-pxdim,:], nPatch)
  mag[pydim:YInv-pydim,XInv-pxdim:XInv,:] = np.multiply(mag[pydim:YInv-pydim,XInv-pxdim:XInv,:], nPatch)
  mag[pydim:YInv-pydim,0:pxdim,:] = np.multiply(mag[pydim:YInv-pydim,0:pxdim,:], nPatch)
  return mag

# find input files in the target dir "pathData"
have1=False
have2=False
elapsed=0
n=0
for image, name, level, line, meta in process_sp3d(pathData):
  #outFileHandle=open(trainCSV,'wt')
  #outFileHandle.close()
  print(name, level, image.shape)
  mYCEN = float(meta['YCEN'])
  mXCEN = float(meta['XCEN'])
  mYSCALE = float(meta['YSCALE'])
  mXSCALE = float(meta['XSCALE'])
  #mCRPIX2 = meta['CRPIX2']
  #mFOVY = meta['FOVY']
  #mCRVAL2 = meta['CRVAL2']
  #mFOVX = meta['FOVX']
  #print("(y1,x1)=(%f,%f)"%(mYCEN+FOVY/2,mXCEN+FOVX/2))
  #print("(y2,x2)=(%f,%f)"%(mYCEN-FOVY/2,mXCEN-FOVX/2))
  print("(yc,xc)=(%f,%f)"%(mYCEN,mXCEN))
  print("(y1,x1)=(%f,%f)"%(mYCEN+YDim/2*mYSCALE,mXCEN+XDim/2*mXSCALE))
  print("(y2,x2)=(%f,%f)"%(mYCEN-YDim/2*mYSCALE,mXCEN-XDim/2*mXSCALE))
  plt.gray()
  if level==1:
    have1 = True
    level1 = image.copy()
    imLevel1I = image[plotWLi,0:YInv,0:XInv,0].copy()
    imLevel1Q = image[plotWLq,0:YInv,0:XInv,1].copy()
    imLevel1U = image[plotWLu,0:YInv,0:XInv,2].copy()
    imLevel1V = image[plotWLv,0:YInv,0:XInv,3].copy()
  else:
    have2 = True
    level2 = image.copy()
    imLevel2S = image[0,0:YInv,0:XInv,0].copy()
    imLevel2I = image[0,0:YInv,0:XInv,1].copy()
    imLevel2A = image[0,0:YInv,0:XInv,2].copy()

  if have1 and have2:
    have1 = False
    have2 = False
    # forward pass through the model to invert the Level 1 SP images and predict the Level 2 image mag fld maps
    level1 = normalize(level1)
    start = time.time()
    inversion = invert_patch(level1, level2)
    end = time.time()
    imPredictS = inversion[:,:,0]
    imPredictI = inversion[:,:,1]
    imPredictA = inversion[:,:,2]
    elapsed += end - start
    level1 = denormalize(level1)
    n += 1

    #Full field
    #xlim = (0, XInv)
    #ylim = (0, YInv)

    xlim = (250, 450)
    ylim = (150, 350)
    #xlim = (250, 400)
    #ylim = (250, 400)
    #xlim = (140, 360)
    #ylim = (140, 360)
    #ylim = (250, 450)
    #xlim = (450, 650)
    # prepare to visualize
    showColorbar = True
    showTitle = False
    showTicks = False
    colormap = plt.cm.inferno
    #colormap = plt.cm.jet
    #colormap = plt.cm.Spectral_r
    #colormap = plt.cm.gist_rainbow_r
    #colormap = plt.cm.hsv

#'viridis', 'plasma', 'inferno', 'magma', 'cividis'
#'Spectral', 'coolwarm', 'bwr', 'seismic'
#'twilight', 'twilight_shifted', 'hsv'
    if True:
      fig1 = plt.figure(1, figsize=(8,4))
      #fig1.suptitle('%s Mag Field'%(name), fontsize=14)
      plt.subplot(231)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('MERLIN Magnetic Field %s (%s)'%(MagName[0],MagUnit[0]))
      if showTicks == False:
        plt.axis('off')
      norm_s = mpc.Normalize(vmin=0.,vmax=3000.0)
      im = plt.imshow(imLevel2S,cmap=colormap,norm=norm_s)
      plt.subplot(234)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('SPIN Magnetic Field %s (%s)'%(MagName[0],MagUnit[0]))
      if showTicks:
        plt.subplots_adjust(hspace=0.10, wspace=0.10)
      else:
        #plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.axis('off')
      im = plt.imshow(imPredictS,cmap=colormap,norm=norm_s)
      if showColorbar:
        cbar_ax = fig1.add_axes([0.02, 0.05, 0.30, 0.02])
        cb = fig1.colorbar(im, cax=cbar_ax, ticks=np.arange(0.0, 3000.0+1000.0, step=1000.0), orientation='horizontal')
        #cb.set_label(MagUnit[0], fontsize=12)
        cbar_ax.tick_params(labelsize=14)
        #plt.text(1.10, 0.0, MagUnit[0], horizontalalignment='left', verticalalignment='center', transform=cbar_ax.transAxes, fontsize=14)
        
      norm_i = mpc.Normalize(vmin=0.,vmax=180.)
      plt.subplot(232)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('MERLIN Magnetic Field %s (%s)'%(MagName[1],MagUnit[1]))
      if showTicks == False:
        plt.axis('off')
      #else:
      #  plt.yticks(np.arange(0.0, 180.0, step=45.0))
      im = plt.imshow(imLevel2I,cmap=colormap,norm=norm_i)
      plt.subplot(235)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('SPIN Magnetic Field %s (%s)'%(MagName[1],MagUnit[1]))
      if showTicks:
        plt.subplots_adjust(hspace=0.20, wspace=0.12)
      else:
        #plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.axis('off')
      im = plt.imshow(imPredictI,cmap=colormap,norm=norm_i)
      if showColorbar:
        #cbar_ax2 = fig1.add_axes([0.90, 0.29, 0.01, 0.41])
        cbar_ax2 = fig1.add_axes([0.38, 0.05, 0.55, 0.02])
        cb = fig1.colorbar(im, cax=cbar_ax2, ticks=np.arange(0.0, 180.0+45.0, step=45.0), orientation='horizontal')
        #cb.set_label(MagUnit[1], fontsize=12)
        #plt.text(1.10, 0.0, MagUnit[1], horizontalalignment='left', verticalalignment='top', transform=cbar_ax2.transAxes, fontsize=10)
        cbar_ax2.tick_params(labelsize=14)
        #plt.text(1.10, 0.0, MagUnit[1], horizontalalignment='center', verticalalignment='center', fontsize=14, transform=cbar_ax2.transAxes)
        #plt.colorbar(fraction=0.046, pad=0.04, shrink=0.8)
      norm_a = mpc.Normalize(vmin=0.,vmax=180.)
      plt.subplot(233)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('MERLIN Magnetic Field %s (%s)'%(MagName[2],MagUnit[2]))
      if showTicks == False:
        plt.axis('off')
      else:
        plt.yticks(np.arange(0.0, 180.0, step=45.0))
      im = plt.imshow(imLevel2A,cmap=colormap,norm=norm_a)
      #if showColorbar:
        #plt.colorbar(fraction=0.046, pad=0.04, shrink=0.8)
      plt.subplot(236)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('SPIN Magnetic Field %s (%s)'%(MagName[2],MagUnit[2]))
      if showTicks:
        plt.subplots_adjust(hspace=0.20, wspace=0.12)
      else:
        #plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.axis('off')
      im = plt.imshow(imPredictA,cmap=colormap,norm=norm_a)
      #if showColorbar:
      #  cbar_ax = fig1.add_axes([0.92, 0.29, 0.01, 0.41])
      #  cb = fig1.colorbar(im, cax=cbar_ax)
      #  cb.set_label(MagUnit[2])
      #plt.tight_layout()
      plt.subplots_adjust(left=0.02, bottom=0.10, right=0.98, top=0.98, hspace=0.10, wspace=0.10)
      plt.show()
      plt.close(fig1)

    if False:
      fig1 = plt.figure(1)
      fig1.suptitle('%s Hinode SOT SP'%(name), fontsize=14)
      #plt.subplot(121)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('%.2fA Stokes %s'%(line, SPName[0]))
      if showTicks == False:
        plt.axis('off')
      #norm_spi = mpc.Normalize(vmin=0.0,vmax=roundup(imLevel1I.max(),3))
      norm_spi = mpc.Normalize(vmin=0.0,vmax=15000.0)
      im = plt.imshow(imLevel1I,cmap=colormap,norm=norm_spi)
      if showColorbar:
        cbar_ax = fig1.add_axes([0.82, 0.29, 0.01, 0.41])
        cb = fig1.colorbar(im, cax=cbar_ax)
        cb.set_label("Counts")
      #plt.subplot(122)
      fig2 = plt.figure(2)
      fig2.suptitle('%s Hinode SOT SP'%(name), fontsize=14)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('%.2fA Stokes %s'%(line, SPName[1]))
      if showTicks:
        plt.subplots_adjust(hspace=0.20, wspace=0.27)
      else:
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.axis('off')
      #norm_spq = mpc.Normalize(vmin=rounddown(imLevel1Q.min(),3),vmax=roundup(imLevel1Q.max(),3))
      norm_spq = mpc.Normalize(vmin=-3000.0,vmax=3000.0)
      im = plt.imshow(imLevel1Q,cmap=colormap,norm=norm_spq)
      if showColorbar:
        cbar_ax = fig2.add_axes([0.82, 0.29, 0.01, 0.41])
        cb = fig2.colorbar(im, cax=cbar_ax)
        cb.set_label("Counts")
      fig3 = plt.figure(3)
      fig3.suptitle('%s Hinode SOT SP'%(name), fontsize=14)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('%.2fA Stokes %s'%(line, SPName[2]))
      if showTicks:
        plt.subplots_adjust(hspace=0.20, wspace=0.27)
      else:
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.axis('off')
      #norm_spu = mpc.Normalize(vmin=rounddown(imLevel1U.min(),3),vmax=roundup(imLevel1U.max(),3))
      norm_spu = mpc.Normalize(vmin=-3000.0,vmax=3000.0)
      im = plt.imshow(imLevel1U,cmap=colormap,norm=norm_spu)
      if showColorbar:
        cbar_ax = fig3.add_axes([0.82, 0.29, 0.01, 0.41])
        cb = fig3.colorbar(im, cax=cbar_ax)
        cb.set_label("Counts")
      fig4 = plt.figure(4)
      fig4.suptitle('%s Hinode SOT SP'%(name), fontsize=14)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('%.2fA Stokes %s'%(line, SPName[3]))
      if showTicks:
        plt.subplots_adjust(hspace=0.20, wspace=0.27)
      else:
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.axis('off')
      #norm_spv = mpc.Normalize(vmin=rounddown(imLevel1V.min(),3),vmax=roundup(imLevel1V.max(),3))
      norm_spv = mpc.Normalize(vmin=-3000.0,vmax=3000.0)
      im = plt.imshow(imLevel1V,cmap=colormap,norm=norm_spv)
      if showColorbar:
        cbar_ax = fig4.add_axes([0.82, 0.29, 0.01, 0.41])
        cb = fig4.colorbar(im, cax=cbar_ax)
        cb.set_label("Counts")
      fig5 = plt.figure(5)
      fig5.suptitle('%s Mag Field'%(name), fontsize=14)
      plt.subplot(121)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('MERLIN Magnetic Field %s (%s)'%(MagName[0],MagUnit[0]))
      if showTicks == False:
        plt.axis('off')
      #norm_s = mpc.Normalize(vmin=0.,vmax=roundup(imLevel2S.max(),3))
      norm_s = mpc.Normalize(vmin=0.,vmax=3000.0)
      #cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
      #cpick.set_array([])
      #colormap = blue_red_colormap(0.,4000.)
      im = plt.imshow(imLevel2S,cmap=colormap,norm=norm_s)
      #if showColorbar:
      #  plt.colorbar(fraction=0.046, pad=0.04, shrink=0.8)
      plt.subplot(122)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('SPIN Magnetic Field %s (%s)'%(MagName[0],MagUnit[0]))
      if showTicks:
        plt.subplots_adjust(hspace=0.20, wspace=0.15)
      else:
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.axis('off')
      im = plt.imshow(imPredictS,cmap=colormap,norm=norm_s)
      if showColorbar:
        cbar_ax = fig5.add_axes([0.92, 0.29, 0.01, 0.41])
        cb = fig5.colorbar(im, cax=cbar_ax)
        cb.set_label(MagUnit[0])
      fig6 = plt.figure(6)
      fig6.suptitle('%s Magnetic Field'%(name), fontsize=14)
      norm_i = mpc.Normalize(vmin=0.,vmax=180.)
      plt.subplot(121)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('MERLIN Magnetic Field %s (%s)'%(MagName[1],MagUnit[1]))
      if showTicks == False:
        plt.axis('off')
      im = plt.imshow(imLevel2I,cmap=colormap,norm=norm_i)
      #if showColorbar:
      #  plt.colorbar(fraction=0.046, pad=0.04, shrink=0.8)
      plt.subplot(122)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('SPIN Magnetic Field %s (%s)'%(MagName[1],MagUnit[1]))
      if showTicks:
        plt.subplots_adjust(hspace=0.20, wspace=0.15)
      else:
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.axis('off')
      im = plt.imshow(imPredictI,cmap=colormap,norm=norm_i)
      if showColorbar:
        cbar_ax = fig6.add_axes([0.92, 0.29, 0.01, 0.41])
        cb = fig6.colorbar(im, cax=cbar_ax)
        cb.set_label(MagUnit[1])
        #plt.colorbar(fraction=0.046, pad=0.04, shrink=0.8)
      fig7 = plt.figure(7)
      fig7.suptitle('%s Magnetic Field'%(name), fontsize=14)
      norm_a = mpc.Normalize(vmin=0.,vmax=180.)
      plt.subplot(121)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('MERLIN Magnetic Field %s (%s)'%(MagName[2],MagUnit[2]))
      if showTicks == False:
        plt.axis('off')
      im = plt.imshow(imLevel2A,cmap=colormap,norm=norm_a)
      #if showColorbar:
        #plt.colorbar(fraction=0.046, pad=0.04, shrink=0.8)
      plt.subplot(122)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('SPIN Magnetic Field %s (%s)'%(MagName[2],MagUnit[2]))
      if showTicks:
        plt.subplots_adjust(hspace=0.20, wspace=0.15)
      else:
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.axis('off')
      im = plt.imshow(imPredictA,cmap=colormap,norm=norm_a)
      if showColorbar:
        cbar_ax = fig7.add_axes([0.92, 0.29, 0.01, 0.41])
        cb = fig7.colorbar(im, cax=cbar_ax)
        cb.set_label(MagUnit[2])
      plt.show()
      plt.close(fig1)
      plt.close(fig2)
      plt.close(fig3)
      plt.close(fig4)
      plt.close(fig5)
      plt.close(fig6)
      plt.close(fig7)

    if False:
      fig2 = plt.figure(2)
      fig2.suptitle('%s Sections'%(name), fontsize=14)
      plt.subplot(121)
      yi = int(np.sum(ylim) / 2)
      sx_x = imLevel2S[yi][xlim[0]:xlim[1]]
      sx_y = imPredictS[yi][xlim[0]:xlim[1]]
      sx = np.linspace(xlim[0], xlim[1], sx_x.size)
      plt.title('%s Y=%d '%(MagName[0], yi))
      plt.plot(sx, sx_y, color='magenta', linestyle="solid")
      plt.plot(sx, sx_x, color='green', linestyle="dotted")
      plt.xlim(xlim)
      plt.xlabel("X")
      plt.ylabel("kG")
      plt.subplot(122)
      xi = int(np.sum(xlim) / 2)
      sy_x = imLevel2S[ylim[0]:ylim[1],xi]
      sy_y = imPredictS[ylim[0]:ylim[1],xi]
      sy = np.linspace(ylim[0], ylim[1], sy_x.size)
      plt.title('%s X=%d '%(MagName[0], xi))
      plt.plot(sy, sy_y, color='magenta', linestyle="solid")
      plt.plot(sy, sy_x, color='green', linestyle="dotted")
      plt.xlim(ylim)
      plt.xlabel("Y")
      plt.ylabel("kG")

      fig3 = plt.figure(3)
      fig3.suptitle('%s Sections'%(name), fontsize=14)
      plt.subplot(121)
      yi = int(np.sum(ylim) / 2)
      sx_x = imLevel2I[yi][xlim[0]:xlim[1]]
      sx_y = imPredictI[yi][xlim[0]:xlim[1]]
      sx = np.linspace(xlim[0], xlim[1], sx_x.size)
      plt.title('%s Y=%d '%(MagName[1], yi))
      plt.plot(sx, sx_y, color='magenta', linestyle="solid")
      plt.plot(sx, sx_x, color='green', linestyle="dotted")
      plt.xlim(xlim)
      plt.xlabel("X")
      plt.ylabel("deg")
      plt.subplot(122)
      xi = int(np.sum(xlim) / 2)
      sy_x = imLevel2I[ylim[0]:ylim[1],xi]
      sy_y = imPredictI[ylim[0]:ylim[1],xi]
      sy = np.linspace(ylim[0], ylim[1], sy_x.size)
      plt.title('%s X=%d '%(MagName[1], xi))
      plt.plot(sy, sy_y, color='magenta', linestyle="solid")
      plt.plot(sy, sy_x, color='green', linestyle="dotted")
      plt.xlim(ylim)
      plt.xlabel("Y")
      plt.ylabel("deg")

      fig4 = plt.figure(4)
      fig4.suptitle('%s Sections'%(name), fontsize=14)
      plt.subplot(121)
      yi = int(np.sum(ylim) / 2)
      sx_x = imLevel2A[yi][xlim[0]:xlim[1]]
      sx_y = imPredictA[yi][xlim[0]:xlim[1]]
      sx = np.linspace(xlim[0], xlim[1], sx_x.size)
      #plt.ylim(ylim)
      plt.title('%s Y=%d '%(MagName[2], yi))
      plt.plot(sx, sx_y, color='magenta', linestyle="solid")
      plt.plot(sx, sx_x, color='green', linestyle="dotted")
      plt.xlim(xlim)
      plt.xlabel("X")
      plt.ylabel("deg")
      plt.subplot(122)
      xi = int(np.sum(xlim) / 2)
      sy_x = imLevel2A[ylim[0]:ylim[1],xi]
      sy_y = imPredictA[ylim[0]:ylim[1],xi]
      sy = np.linspace(ylim[0], ylim[1], sy_x.size)
      #plt.ylim(ylim)
      plt.title('%s X=%d '%(MagName[2], xi))
      plt.plot(sy, sy_y, color='magenta', linestyle="solid")
      plt.plot(sy, sy_x, color='green', linestyle="dotted")
      plt.xlim(ylim)
      plt.xlabel("Y")
      plt.ylabel("deg")

      plt.subplots_adjust(hspace=0.05, wspace=0.30)
      plt.show()
      plt.close(fig2)
      plt.close(fig3)
      plt.close(fig4)
      plt.close(fig4)

print('%d test examples in %.1f seconds: %.3f avg.'%(n,elapsed, elapsed/n))
