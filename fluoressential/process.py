import time

import numpy as np
from scipy import ndimage as ndi
from skimage import img_as_float
from skimage.io import imread
from skimage.measure import regionprops
from skimage.filters import (threshold_local, gaussian,
    laplace, threshold_otsu, sobel, median, threshold_li)
from skimage.morphology import disk, binary_erosion, remove_small_objects, remove_small_holes
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.segmentation import clear_border


def preprocess_img(imgf, bkg_imgf=None, offset=10):
    """Subtract background and denoise fluorescence image."""
    img = img_as_float(imread(imgf))
    sig = estimate_sigma(img)
    img = denoise_nl_means(img, h=sig, sigma=sig, patch_size=9, patch_distance=7)
    if bkg_imgf:
        bkg = img_as_float(imread(bkg_imgf))
    else:
        bkg = threshold_local(img, block_size=1001, offset=offset*sig, method='gaussian')
    img = img - bkg
    img[img < 0] = 0
    return img, sig


def segment_object(imgf, bkg_imgf=None, min_size=5000, fill_size=5000, clbr=False, offset=10):
    """Segment out bright objects from fluorescence images."""
    img, sig = preprocess_img(imgf, bkg_imgf, offset)
    thr = median(img > threshold_li(img))
    thr = remove_small_objects(ndi.label(thr)[0].astype(bool), min_size=min_size)
    thr = remove_small_holes(thr.astype(bool), area_threshold=fill_size)
    if clbr:
        thr = clear_border(thr, buffer_size=5)
    return thr, img


def segment_clusters(imgf, bkg_imgf=None, offset=10):
    """Segment out bright clusters from fluorescence images."""
    img, sig = preprocess_img(imgf, bkg_imgf, offset)
    log = laplace(gaussian(img, sigma=1.5))
    bkg = threshold_local(log, block_size=25, offset=-2*sig, method='gaussian')
    thr = log > bkg
    return thr, img
