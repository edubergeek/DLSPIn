# spin
## HINODE SpectroPolarimetric inversion
Authors: Stanley Dodds

Affiliations: Institute for Astronomy, Univ. of Hawaii Manoa

Date: 2018-10-13

## Data Description
URL: 
Citation: 

## Level 2 FITS file
Each FITS file is a spectral inversion using the MERLIN code implementation of Milne-Eddington inversion from the level 1 Stokes params

The Y (512) dimension is Y on the solar disc in 0.317 arcsec pixels.

The X (112) dimension is X on the solar disc in #### arcsec pixels.

The Z (3) dimension are the magnetic field strength, inclination and azimuth.
0 Field Strength
1 Field Inclination
2 Field Azimuth

## Level 1 tarball
Each tar file contains 875 FITS files.
Each FITS file is a point in time slit image on the solar disc in x arcsec pixels.

The Y (512) dimension is Y on the solar disc in 0.317 arcsec pixels.

The W (112) dimension is wavelength in 0.021549 Angstrom pixels centered (pixel 56.5) on 6302.08A.

The Z (4) dimension is the Stokes vector.
0 I
1 Q
2 U
3 V

The 875 FITS files comprises the X (875) dimension.

## Visualization image data
The X (875) dimension is X on the solar disc where x=56 (6302A) for each of 875 FITS files.

The Y (512) dimension is Y on the solar disc for all y in Y.

The Z (1) dimension is the Stokes I parameter (z=0).

## Training image data
The X (875) dimension is X on the solar disc where x=56 (6302A) for each of 875 FITS files.

The Y (512) dimension is Y on the solar disc in 0.317 arcsec pixels.

The Z (4) dimension is the Stokes vector.

## USAGE
prep.py - Load and visualize (plot) each FITS file
Edit parameters at the top of prep.py to set:
  basePath = Path to scan for SP3D Level 1 and 2 images in FITS format
  spNum is the Stokes paramater to use 0123 => IQUV
  wlOffset = Offset in pxiels from the central line

fitsinfo.py - Print out the FITS header values
Edit parameters at the top of fitsinfo.py to set:
  basePath = Path to scan for FITS files

hinode2tfr.py - Convert raw FITS files to TensorFlow TFRecord format and split into train, valid and test sets
Outputs 3 files:
  ../data/train.tfr
  ../data/val.tfr
  ../data/test.tfr
Edit parameters at the top of hinode2tfr.py to set:
  basePath = Path to scan for SP3D Level 1 and 2 images in FITS format
  wlOffset = Offset in pxiels from the central line

train.py - Train a model using training data in TFRecord format
Inputs 2 files:
  ../data/train.tfr
  ../data/val.tfr
Outputs the "best" model as ../data/test1.h5
Edit parameters at the top of train.py to set:
  sizeBatch is the batch size
  nEpochs is the number of epochs to train

invert.py - Invert arbitrary size Hinode images using sliding windows of 64x64 patches
Input directory containing Hinode SOT SP FITS files (level 1)
  ./invert
Outputs magnetograms for each level 1 file found in the input directory
Edit image position and size parameters in invert.py
Optionally plot kernel density estimates
Optionally convert cartesian to spherical magnetic field coordinates
Model,Version,Checkpoint control which trained SPIN model to use.
nPatch=2 use sliding window (64/2 = 32 pixels); nPatch=1 no sliding windows but faster
plotWL[iquv] determines which wavelength bin to plot
Around line 797 
xlim, ylim control the crop rectangle to plot
showColorbar, showTitle, showTicks controls plotting outputs
colormap controls the plot color scheme

## FUTURE WORK

run inference on 2 Hinode samples from SICON paper (were they held out of training? validation?)

