import os, tempfile
import urllib.request
import urllib.request, urllib.error, urllib.parse

import numpy as np
from scipy.misc import imread

from cs231n.fast_layers import conv_forward_fast


"""
Utility functions used for viewing and processing images.
"""


def blur_image(X):
  """
  A very gentle image blurring operation, to be used as a regularizer for image
  generation.

  Inputs:
  - X: Image data of shape (N, 3, H, W)

  Returns:
  - X_blur: Blurred version of X, of shape (N, 3, H, W)
  """
  w_blur = np.zeros((3, 3, 3, 3))
  b_blur = np.zeros(3)
  blur_param = {'stride': 1, 'pad': 1}
  for i in range(3):
    w_blur[i, i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]], dtype=np.float32)
  w_blur /= 200.0
  return conv_forward_fast(X, w_blur, b_blur, blur_param)[0]


SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
def preprocess_image(img, mean_img=None, mean='image'):
  """
  Convert to float, transepose, and subtract mean pixel

  Input:
  - img: (H, W, 3)

  Returns:
  - (1, 3, H, 3)
  """
  if mean_img is None:
    return (img.astype(np.float32)/255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD
  if mean == 'image':
    mean = mean_img
  elif mean == 'pixel':
    mean = mean_img.mean(axis=(1, 2), keepdims=True)
  elif mean == 'none':
    mean = 0
  else:
    raise ValueError('mean must be image or pixel or none')
  return img.astype(np.float32).transpose(2, 0, 1)[None] - mean


def deprocess_image(img, mean_img=None, mean='image', renorm=False):
  """
  Add mean pixel, transpose, and convert to uint8

  Input:
  - (1, 3, H, W) or (3, H, W)

  Returns:
  - (H, W, 3)
  """
  if mean == 'image':
    mean = mean_img
  elif mean == 'pixel':
    mean = mean_img.mean(axis=(1, 2), keepdims=True)
  elif mean == 'none':
    mean = 0
  else:
    raise ValueError('mean must be image or pixel or none')
  if img.ndim == 3:
    img = img[None]
  img = (img + mean)[0].transpose(1, 2, 0)
  if renorm:
    low, high = img.min(), img.max()
    img = 255.0 * (img - low) / (high - low)
  return img.astype(np.uint8)

def deprocess_image2018(img, rescale=False):
    """Undo preprocessing on an image and convert back to uint8."""
    img = (img * SQUEEZENET_STD + SQUEEZENET_MEAN)
    if rescale:
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin)
    return np.clip(255 * img, 0.0, 255.0).astype(np.uint8)

def image_from_url(url):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    We write the image to a temporary file then read it back. Kinda gross.
    """
    try:
        f = urllib.request.urlopen(url)
        _, fname = tempfile.mkstemp()
        print('fname:', fname)
        with open(fname, 'wb') as ff:
            ff.write(f.read())
        img = imread(fname)
        try:
            os.remove(fname)
        except:
            print('remove {} error!'.format(fname))
        return img
    except urllib.error.URLError as e:
        print('URL Error: ', e.reason, url)
    except urllib.error.HTTPError as e:
        print('HTTP Error: ', e.code, url)


def load_image(filename, size=None):
    """Load and resize an image from disk.

    Inputs:
    - filename: path to file
    - size: size of shortest dimension after rescaling
    """
    img = imread(filename)
    if size is not None:
        orig_shape = np.array(img.shape[:2])
        min_idx = np.argmin(orig_shape)
        scale_factor = float(size) / orig_shape[min_idx]
        new_shape = (orig_shape * scale_factor).astype(int)
        img = imresize(img, scale_factor)
    return img
