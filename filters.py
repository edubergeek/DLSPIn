import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
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
Version='5-3'
Checkpoint='187'
Basename="%s-v%s"%(Model,Version)
doWLConvolution=True

IMG_DIM = 3

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

pathData='/d/hinode/spin/invert/20061114_163005'
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

plotWL=24
plotPatch=2*PX+3

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
 
# evaluate loaded model on test data
model.compile(optimizer=Adam(lr=1e-3), loss='mae', metrics=['mae','mse'])

# global plot settings
Nr = 2
Nc = 2

def plot_normalize(arr):
  arr_min = np.min(arr)
  return (arr-arr_min)/(np.max(arr)-arr_min)

def plot_histogram(values):
  n, bins, patches = plt.hist(values.reshape(-1), 50, normed=1)
  bin_centers = 0.5 * (bins[:-1] + bins[1:])

  for c, p in zip(normalize(bin_centers), patches):
    plt.setp(p, 'facecolor', cm.viridis(c))

  plt.show()

def explode(data):
  shape_arr = np.array(data.shape)
  size = shape_arr[:3]*2 - 1
  exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
  exploded[::2, ::2, ::2] = data
  return exploded

def expand_coordinates(indices):
  x, y, z = indices
  x[1::2, :, :] += 1
  y[:, 1::2, :] += 1
  z[:, :, 1::2] += 1
  return x, y, z

def plot_cube(ax, cube, angle=66):
  cube = plot_normalize(cube)

  facecolors = cm.inferno(cube)
  facecolors[:,:,:,-1] = cube  # alpha channel = voxel value
  facecolors = explode(facecolors)

  filled = abs(facecolors[:,:,:,-1]) >= 0.6
  z, y, x = expand_coordinates(np.indices(np.array(filled.shape) + 1))

  ax.view_init(60, angle)
  ax.set_xlim(right=IMG_DIM*2)
  ax.set_ylim(top=IMG_DIM*2)
  ax.set_zlim(top=IMG_DIM*2)

  ax.set_xlabel('$X$') #, fontsize=20)
  ax.set_ylabel('$Y$')
  #ax.yaxis._axinfo['label']['space_factor'] = 3.0
  # set z ticks and labels
  #ax.set_zticks([-2, 0, 2])
  # change fontsize
  #for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(10)
  # disable auto rotation
  ax.zaxis.set_rotate_label(False) 
  ax.set_zlabel('$\lambda$', rotation = 0) #, fontsize=30)

  im = ax.voxels(x, y, z, filled, facecolors=facecolors)
  #plot.show()

def plot_filters(filters, title):
  # normalize filter values to 0-1 so we can visualize them
  f_min, f_max = filters.min(), filters.max()
  #filters = (filters - f_min) / (f_max - f_min)
  # plot first few filters
  n_filters = 8
  # bounds check on filters based on filters.shape
  for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, :, i]
    fig = plt.figure(figsize=(20/2.54, 20/2.54))# originally 30
    fig.suptitle(title%(i), fontsize=14)
    ax = fig.gca(projection='3d')

    # bounds check on filters based on filters.shape
    for j in range(8):
      # specify subplot and turn of axis
      #ax = plt.subplot(n_filters, 3, ix)
      #ax.set_xticks([])
      #ax.set_yticks([])
      # plot filter channel in grayscale
      #plt.imshow(f[1, :, :, j], cmap='gray')

      # plot filter channel in 3D
      #ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5);
      #ax = plt.subplot(n_filters, int(j/2)+1, int(j%4)+1, projection='3d')
      #ax = fig.add_subplot(800+(int(j/4)+1)*10+int(j%4)+1, projection='3d')
      #ax = fig.gca(projection='3d')
      if j == 0:
        ax.remove()
      ax = fig.add_subplot(3, 3, j+1, projection='3d')
      plot_cube(ax, f[:, :, :, j])

    # show the filters for this input
    norm = mpc.Normalize(vmin=f_min, vmax=f_max)
    m = cm.ScalarMappable(cmap=plt.cm.inferno, norm=norm)
    m.set_array([])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.10, 0.02, 0.80])
    cb = fig.colorbar(m, cax=cbar_ax)
    cb.set_label("Weight")

    plt.show()

plotFilters = "conv3d_7"
# summarize filter shapes
for layer in model.layers:
  # check for convolutional layer
  if 'conv' not in layer.name:
    continue
  # get filter weights
  filters, biases = layer.get_weights()
  print(layer.name, filters.shape)
  print("filters shape", filters.shape)
  #plot_histogram(arr)

  if layer.name == plotFilters:
    title='Layer %s Filters'%layer.name
    plot_filters(filters, title+' - Input %d')

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

  if False and plotPatch < patchesPerImage:
    plt.imshow(batch[plotPatch,plotWL,:,:,0])
    plt.show()
    plt.imshow(batch[plotPatch,plotWL,:,:,1])
    plt.show()
    plt.imshow(batch[plotPatch,plotWL,:,:,2])
    plt.show()
    plt.imshow(batch[plotPatch,plotWL,:,:,3])
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

      if True and n == plotPatch:
        colormap = plt.cm.inferno
        fig1 = plt.figure(1)
        fig1.suptitle('%s Hinode SOT SP'%(name), fontsize=14)
        plt.subplot(221)
        plt.title("Patch %d SOT SP-I"%(yi * PX + xi))
        plt.imshow(SP[plotWL,y:y+PYDim,x:x+PXDim,0], cmap=colormap)
        plt.subplot(222)
        plt.title("Patch %d SOT SP-Q"%(yi * PX + xi))
        plt.imshow(SP[plotWL,y:y+PYDim,x:x+PXDim,1], cmap=colormap)
        plt.subplot(223)
        plt.title("Patch %d SOT SP-U"%(yi * PX + xi))
        plt.imshow(SP[plotWL,y:y+PYDim,x:x+PXDim,2], cmap=colormap)
        plt.subplot(224)
        plt.title("Patch %d SOT SP-V"%(yi * PX + xi))
        plt.imshow(SP[plotWL,y:y+PYDim,x:x+PXDim,3], cmap=colormap)
        plt.subplots_adjust(hspace=0.15, wspace=0.15)

        fig2 = plt.figure(2)
        fig2.suptitle('%s Magnetograms MERLIN vs SPIN'%(name), fontsize=14)
        plt.subplot(231)
        plt.title("Patch %d MERLIN MF-St"%(yi * PX + xi))
        plt.imshow(MF[0,y:y+PYDim,x:x+PXDim,0], cmap=colormap)
        plt.subplot(234)
        plt.title("Patch %d SPIN MF-St"%(yi * PX + xi))
        plt.imshow(inversion[n,:,:,0], cmap=colormap)
        plt.subplot(232)
        plt.title("Patch %d MERLIN MF-Inc"%(yi * PX + xi))
        plt.imshow(MF[0,y:y+PYDim,x:x+PXDim,1], cmap=colormap)
        plt.subplot(235)
        plt.title("Patch %d SPIN MF-Inc"%(yi * PX + xi))
        plt.imshow(inversion[n,:,:,1], cmap=colormap)
        plt.subplot(233)
        plt.title("Patch %d MERLIN MF-Az"%(yi * PX + xi))
        plt.imshow(MF[0,y:y+PYDim,x:x+PXDim,2], cmap=colormap)
        plt.subplot(236)
        plt.title("Patch %d SPIN MF-Az"%(yi * PX + xi))
        plt.imshow(inversion[n,:,:,2], cmap=colormap)
        plt.subplots_adjust(hspace=0.15, wspace=0.15)
        plt.show()
        plt.close(fig1)
        plt.close(fig2)
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
    WL = int(np.ceil(WInv/2))
    imLevel1I = image[WL,0:YInv,0:XInv,0].copy()
    imLevel1Q = image[WL,0:YInv,0:XInv,1].copy()
    imLevel1U = image[WL,0:YInv,0:XInv,2].copy()
    imLevel1V = image[WL,0:YInv,0:XInv,3].copy()
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
    start = time.time()
    level1 = normalize(level1)
    inversion = invert_patch(level1, level2)
    imPredictS = inversion[:,:,0]
    imPredictI = inversion[:,:,1]
    imPredictA = inversion[:,:,2]
    end = time.time()
    elapsed += end - start
    n += 1

    xlim = (0, XInv)
    ylim = (0, YInv)
    #xlim = (250, 400)
    #ylim = (250, 400)
    #xlim = (140, 360)
    #ylim = (140, 360)
    #xlim = (250, 450)
    #ylim = (150, 350)
    #ylim = (250, 450)
    #xlim = (450, 650)
    # prepare to visualize
    showColorbar = True
    showTitle = True
    showTicks = True
    colormap = plt.cm.inferno
    #colormap = plt.cm.jet
    #colormap = plt.cm.Spectral_r
    #colormap = plt.cm.gist_rainbow_r
    #colormap = plt.cm.hsv

#'viridis', 'plasma', 'inferno', 'magma', 'cividis'
#'Spectral', 'coolwarm', 'bwr', 'seismic'
#'twilight', 'twilight_shifted', 'hsv'
    if True:
      fig1 = plt.figure(1)
      fig1.suptitle('%s Hinode SOT SP'%(name), fontsize=14)
      plt.subplot(121)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('%.2fA Stokes %s'%(line, SPName[0]))
      if showTicks == False:
        plt.axis('off')
      im = plt.imshow(imLevel1I,cmap=colormap)
      #if showColorbar:
      #  plt.colorbar(im, fraction=0.01, pad=0.15)
      plt.subplot(122)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('%.2fA Stokes %s'%(line, SPName[3]))
      if showTicks:
        plt.subplots_adjust(hspace=0.15, wspace=0.30)
      else:
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.axis('off')
      plt.imshow(imLevel1V,cmap=colormap)
      if showColorbar:
        plt.colorbar(fraction=0.15, pad=0.25)
      fig2 = plt.figure(2)
      fig2.suptitle('%s Mag Field'%(name), fontsize=14)
      plt.subplot(121)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('MERLIN Magnetic Field %s (%s)'%(MagName[0],MagUnit[0]))
      if showTicks == False:
        plt.axis('off')
      norm_s = mpc.Normalize(vmin=0.,vmax=4000.)
      #cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
      #cpick.set_array([])
      #colormap = blue_red_colormap(0.,4000.)
      plt.imshow(imLevel2S,cmap=colormap,norm=norm_s)
      if showColorbar:
        plt.colorbar(fraction=0.046, pad=0.04)
      plt.subplot(122)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('SPIN Magnetic Field %s (%s)'%(MagName[0],MagUnit[0]))
      if showTicks:
        plt.subplots_adjust(hspace=0.15, wspace=0.30)
      else:
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.axis('off')
      plt.imshow(imPredictS,cmap=colormap,norm=norm_s)
      if showColorbar:
        plt.colorbar(fraction=0.046, pad=0.04)
      fig3 = plt.figure(3)
      fig3.suptitle('%s Magnetic Field'%(name), fontsize=14)
      norm_i = mpc.Normalize(vmin=0.,vmax=180.)
      plt.subplot(121)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('MERLIN Magnetic Field %s (%s)'%(MagName[1],MagUnit[1]))
      if showTicks == False:
        plt.axis('off')
      plt.imshow(imLevel2I,cmap=colormap,norm=norm_i)
      if showColorbar:
        plt.colorbar(fraction=0.046, pad=0.04)
      plt.subplot(122)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('SPIN Magnetic Field %s (%s)'%(MagName[1],MagUnit[1]))
      if showTicks:
        plt.subplots_adjust(hspace=0.15, wspace=0.30)
      else:
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.axis('off')
      plt.imshow(imPredictI,cmap=colormap,norm=norm_i)
      if showColorbar:
        plt.colorbar(fraction=0.046, pad=0.04)
      plt.subplots_adjust(hspace=0.05, wspace=0.05)
      fig4 = plt.figure(4)
      fig4.suptitle('%s Magnetic Field'%(name), fontsize=14)
      norm_a = mpc.Normalize(vmin=0.,vmax=180.)
      plt.subplot(121)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('MERLIN Magnetic Field %s (%s)'%(MagName[2],MagUnit[2]))
      if showTicks == False:
        plt.axis('off')
      plt.imshow(imLevel2A,cmap=colormap,norm=norm_a)
      if showColorbar:
        plt.colorbar(fraction=0.046, pad=0.04)
      plt.subplot(122)
      plt.ylim(ylim)
      plt.xlim(xlim)
      if showTitle:
        plt.title('SPIN Magnetic Field %s (%s)'%(MagName[2],MagUnit[2]))
      if showTicks:
        plt.subplots_adjust(hspace=0.15, wspace=0.30)
      else:
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.axis('off')
      plt.imshow(imPredictA,cmap=colormap,norm=norm_a)
      if showColorbar:
        #plt.colorbar(fraction=0.046, pad=0.04)
        plt.colorbar(fraction=0.15, pad=0.25)
      plt.subplots_adjust(hspace=0.05, wspace=0.30)
      plt.show()
      plt.close(fig1)
      plt.close(fig2)
      plt.close(fig3)
      plt.close(fig4)

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

print('%d test examples in %.1f seconds: %.3f avg.'%(n,elapsed, elapsed/n))
