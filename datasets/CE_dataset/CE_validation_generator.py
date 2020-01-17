# -*- coding: utf-8 -*-
# This code is adapted from the https://github.com/hendrycks/robustness.
# ==========================================================================================
# NAVER’s modifications are Copyright 2020 NAVER corp. All rights reserved.
# ==========================================================================================

import os
from PIL import Image
import os.path
import time
import numpy as np
import PIL
from multiprocessing import Pool
import argparse

# RESIZE_SIZE = 256
# CROP_SIZE = 256

parser = argparse.ArgumentParser()

parser.add_argument('-v', '--validation_dir', type=str, default=None, help='Validation data directory.')
parser.add_argument('-o', '--output_dir', type=str, default=None, help='Output data directory.')
parser.add_argument('-f', '--frost_dir', type=str, default='./frost', help='frost img file directory.')
parser.add_argument('--num_workers', type=int, default=20, help='Number of processes to preprocess the images.')
parser.add_argument('--RESIZE_SIZE', type=int, default=256, help='Resize size')
parser.add_argument('--CROP_SIZE', type=int, default=224, help='Center crop size')

args = parser.parse_args()
# /////////////// Data Loader ///////////////


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.gif']


def is_image_file(filename):
  """Checks if a file is an image.
  Args:
    filename (string): path to a file
  Returns:
    bool: True if the filename ends with a known image extension
  """
  filename_lower = filename.lower()
  return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def make_dataset(directory):
  images = []
  directory = os.path.expanduser(directory)
  for target in sorted(os.listdir(directory)):
    d = os.path.join(directory, target)
    if not os.path.isdir(d):
      continue

    for root, _, fnames in sorted(os.walk(d)):
      for fname in sorted(fnames):
        if is_image_file(fname):
          path = os.path.join(root, fname)
          item = (path, target)
          images.append(item)

  return images


def count_dataset(directory):
  img_cnt = 0
  directory = os.path.expanduser(directory)
  for target in sorted(os.listdir(directory)):
    d = os.path.join(directory, target)
    if not os.path.isdir(d):
      continue
    
    for root, _, fnames in sorted(os.walk(d)):
      for fname in sorted(fnames):
        if is_image_file(fname):
          img_cnt += 1

  return img_cnt


def pil_loader(path):
  # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')


def default_loader(path):
  return pil_loader(path)


def resize_and_center_crop(img, resize_size, crop_size):
  w = img.size[0]
  h = img.size[1]
  # resize image
  if h > w:
    new_h = int(resize_size * (h/w))
    new_w = resize_size
  else:
    new_h = resize_size
    new_w = int(resize_size * (w/h))
  
  resized_img = img.resize((new_w, new_h), resample=PIL.Image.BILINEAR)
  # crop image
  h_start = int((new_h-crop_size)/2)
  w_start = int((new_w-crop_size)/2)
  cropped_img = resized_img.crop((w_start, h_start,
                                  w_start+crop_size, h_start+crop_size))
  
  return cropped_img


class DistortImageFolder():
  def __init__(self, root, method, severity, start_idx, end_idx,
               transform='imagenet', loader=default_loader):
    imgs = make_dataset(root)
    imgs = imgs[start_idx:end_idx]
    if len(imgs) == 0:
      raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                       "Supported image extensions are: " + ",".join(
        IMG_EXTENSIONS)))

    self.root = root
    self.method = method
    self.severity = severity
    self.imgs = imgs
    self.transform = transform
    self.loader = loader

  def __getitem__(self, index):
    path, target = self.imgs[index]
    img = self.loader(path)
    # default trasformation is set to imagenet preprocessing
    if self.transform == 'imagenet':
      img = resize_and_center_crop(img, resize_size=args.RESIZE_SIZE, crop_size=args.CROP_SIZE)
    img = self.method(img, self.severity)

    save_path = os.path.join(args.output_dir, self.method.__name__,
                             str(self.severity), target)

    if not os.path.exists(save_path):
      os.makedirs(save_path)

    save_path += path[path.rindex('/'):]

    Image.fromarray(np.uint8(img)).save(save_path, quality=85, optimize=True)

    return 0  # we do not care about returning the data

  def __len__(self):
    return len(self.imgs)


# /////////////// Distortion Helpers ///////////////

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings

warnings.simplefilter("ignore", UserWarning)


def auc(errs):  # area under the alteration error curve
  area = 0
  for i in range(1, len(errs)):
    area += (errs[i] + errs[i - 1]) / 2
  area /= len(errs) - 1
  return area


def disk(radius, alias_blur=0.1, dtype=np.float32):
  if radius <= 8:
    L = np.arange(-8, 8 + 1)
    ksize = (3, 3)
  else:
    L = np.arange(-radius, radius + 1)
    ksize = (5, 5)
  X, Y = np.meshgrid(L, L)
  aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
  aliased_disk /= np.sum(aliased_disk)

  # supersample disk to antialias
  return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                        ctypes.c_double,  # radius
                        ctypes.c_double,  # sigma
                        ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
  def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
    wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
  """
  Generate a heightmap using diamond-square algorithm.
  Return square 2d array, side length 'mapsize', of floats in range 0-255.
  'mapsize' must be a power of two.
  """
  assert (mapsize & (mapsize - 1) == 0)
  maparray = np.empty((mapsize, mapsize), dtype=np.float_)
  maparray[0, 0] = 0
  stepsize = mapsize
  wibble = 100

  def wibbledmean(array):
    return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

  def fillsquares():
    """For each square of points stepsize apart,
       calculate middle value as mean of points + wibble"""
    cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
    squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
    squareaccum += np.roll(squareaccum, shift=-1, axis=1)
    maparray[stepsize // 2:mapsize:stepsize,
    stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

  def filldiamonds():
    """For each diamond of points stepsize apart,
       calculate middle value as mean of points + wibble"""
    mapsize = maparray.shape[0]
    drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
    ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
    ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
    lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
    ltsum = ldrsum + lulsum
    maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
    tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
    tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
    ttsum = tdrsum + tulsum
    maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

  while stepsize >= 2:
    fillsquares()
    filldiamonds()
    stepsize //= 2
    wibble /= wibbledecay

  maparray -= maparray.min()
  return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
  h = img.shape[0]
  w = img.shape[1]
  # ceil crop height(= crop width)
  ch = int(np.ceil(h / zoom_factor))
  cw = int(np.ceil(w / zoom_factor))

  top_h = (h - ch) // 2
  top_w = (w - cw) // 2

  img = scizoom(img[top_h:top_h + ch, top_w:top_w + cw], (zoom_factor, zoom_factor, 1), order=1)
  # trim off any extra pixels
  trim_top_h = (img.shape[0] - h) // 2
  trim_top_w = (img.shape[1] - w) // 2

  return img[trim_top_h:trim_top_h + h, trim_top_w:trim_top_w + w]


# /////////////// End Distortion Helpers ///////////////


# /////////////// Distortions ///////////////

def gaussian_noise(x, severity=1):
  c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

  x = np.array(x) / 255.
  return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
  c = [60, 25, 12, 5, 3][severity - 1]

  x = np.array(x) / 255.
  return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
  c = [.03, .06, .09, 0.17, 0.27][severity - 1]

  x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
  return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
  c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

  x = np.array(x) / 255.
  return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity=1):
  c = [1, 2, 3, 4, 6][severity - 1]

  x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
  return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1):
  # sigma, max_delta, iterations
  c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

  x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)

  # locally shuffle pixels
  for i in range(c[2]):
    for h in range(args.CROP_SIZE - c[1], c[1], -1):
      for w in range(args.CROP_SIZE - c[1], c[1], -1):
        dx, dy = np.random.randint(-c[1], c[1], size=(2,))
        h_prime, w_prime = h + dy, w + dx
        # swap
        x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

  return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255


def defocus_blur(x, severity=1):
  c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

  x = np.array(x) / 255.
  kernel = disk(radius=c[0], alias_blur=c[1])

  channels = []
  for d in range(3):
    channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
  channels = np.array(channels).transpose((1, 2, 0))  # 3xCROP_SIZExCROP_SIZE -> CROP_SIZExCROP_SIZEx3

  return np.clip(channels, 0, 1) * 255


def motion_blur(x, severity=1):
  c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

  output = BytesIO()
  x.save(output, format='PNG')
  x = MotionImage(blob=output.getvalue())

  x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

  x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
           cv2.IMREAD_UNCHANGED)

  if x.shape != (args.CROP_SIZE, args.CROP_SIZE):
    return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
  else:  # greyscale to RGB
    return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def zoom_blur(x, severity=1):
  c = [np.arange(1, 1.11, 0.01),
     np.arange(1, 1.16, 0.01),
     np.arange(1, 1.21, 0.02),
     np.arange(1, 1.26, 0.02),
     np.arange(1, 1.31, 0.03)][severity - 1]

  x = (np.array(x) / 255.).astype(np.float32)
  out = np.zeros_like(x)
  for zoom_factor in c:
    out += clipped_zoom(x, zoom_factor)

  x = (x + out) / (len(c) + 1)
  return np.clip(x, 0, 1) * 255


def fog(x, severity=1):
  c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]

  x = np.array(x) / 255.
  max_val = x.max()
  x += c[0] * plasma_fractal(wibbledecay=c[1])[:args.CROP_SIZE, :args.CROP_SIZE][..., np.newaxis]
  return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1):
  c = [(1, 0.4),
     (0.8, 0.6),
     (0.7, 0.7),
     (0.65, 0.7),
     (0.6, 0.75)][severity - 1]
  idx = np.random.randint(5)
  filename = ['frost1.png', 'frost2.png', 'frost3.png', 'frost4.jpg', 'frost5.jpg', 'frost6.jpg'][idx]
  frost = cv2.imread(os.path.join(args.frost_dir, filename))
  # randomly crop and convert to rgb
  x_start, y_start = np.random.randint(0, frost.shape[0] - args.CROP_SIZE), np.random.randint(0, frost.shape[1] - args.CROP_SIZE)
  frost = frost[x_start:x_start + args.CROP_SIZE, y_start:y_start + args.CROP_SIZE][..., [2, 1, 0]]

  return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


def snow(x, severity=1):
  c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
     (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
     (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
     (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
     (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

  x = np.array(x, dtype=np.float32) / 255.
  snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

  snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
  snow_layer[snow_layer < c[3]] = 0

  snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
  output = BytesIO()
  snow_layer.save(output, format='PNG')
  snow_layer = MotionImage(blob=output.getvalue())

  snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

  snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                cv2.IMREAD_UNCHANGED) / 255.
  snow_layer = snow_layer[..., np.newaxis]

  x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(args.CROP_SIZE, args.CROP_SIZE,
                                                                                        1) * 1.5 + 0.5)
  return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def spatter(x, severity=1):
  c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
     (0.65, 0.3, 3, 0.68, 0.6, 0),
     (0.65, 0.3, 2, 0.68, 0.5, 0),
     (0.65, 0.3, 1, 0.65, 1.5, 1),
     (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
  x = np.array(x, dtype=np.float32) / 255.

  liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

  liquid_layer = gaussian(liquid_layer, sigma=c[2])
  liquid_layer[liquid_layer < c[3]] = 0
  if c[5] == 0:
    liquid_layer = (liquid_layer * 255).astype(np.uint8)
    dist = 255 - cv2.Canny(liquid_layer, 50, 150)
    dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
    _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
    dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
    dist = cv2.equalizeHist(dist)
    #   ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
    #   ker -= np.mean(ker)
    ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    dist = cv2.filter2D(dist, cv2.CV_8U, ker)
    dist = cv2.blur(dist, (3, 3)).astype(np.float32)

    m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
    m /= np.max(m, axis=(0, 1))
    m *= c[4]

    # water is pale turqouise
    color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                238 / 255. * np.ones_like(m[..., :1]),
                238 / 255. * np.ones_like(m[..., :1])), axis=2)

    color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

    return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
  else:
    m = np.where(liquid_layer > c[3], 1, 0)
    m = gaussian(m.astype(np.float32), sigma=c[4])
    m[m < 0.8] = 0
    #     m = np.abs(m) ** (1/c[4])

    # mud brown
    color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                42 / 255. * np.ones_like(x[..., :1]),
                20 / 255. * np.ones_like(x[..., :1])), axis=2)

    color *= m[..., np.newaxis]
    x *= (1 - m[..., np.newaxis])

    return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1):
  c = [0.4, .3, .2, .1, .05][severity - 1]

  x = np.array(x) / 255.
  means = np.mean(x, axis=(0, 1), keepdims=True)
  return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
  c = [.1, .2, .3, .4, .5][severity - 1]

  x = np.array(x) / 255.
  x = sk.color.rgb2hsv(x)
  x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
  x = sk.color.hsv2rgb(x)

  return np.clip(x, 0, 1) * 255


def saturate(x, severity=1):
  c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

  x = np.array(x) / 255.
  x = sk.color.rgb2hsv(x)
  x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
  x = sk.color.hsv2rgb(x)

  return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
  c = [25, 18, 15, 10, 7][severity - 1]

  output = BytesIO()
  x.save(output, 'JPEG', quality=c)
  x = PILImage.open(output)

  return x


def pixelate(x, severity=1):
  c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

  x = x.resize((int(args.CROP_SIZE * c), int(args.CROP_SIZE * c)), PILImage.BOX)
  x = x.resize((args.CROP_SIZE, args.CROP_SIZE), PILImage.BOX)

  return x


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity=1):
  c = [(244 * 2, 244 * 0.7, 244 * 0.1),  # CROP_SIZE should have been CROP_SIZE, but ultimately nothing is incorrect
       (244 * 2, 244 * 0.08, 244 * 0.2),
       (244 * 0.05, 244 * 0.01, 244 * 0.02),
       (244 * 0.07, 244 * 0.01, 244 * 0.02),
       (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]

  image = np.array(image, dtype=np.float32) / 255.
  shape = image.shape
  shape_size = shape[:2]

  # random affine
  center_square = np.float32(shape_size) // 2
  square_size = min(shape_size) // 3
  pts1 = np.float32([center_square + square_size,
             [center_square[0] + square_size, center_square[1] - square_size],
             center_square - square_size])
  pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
  M = cv2.getAffineTransform(pts1, pts2)
  image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

  dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
           c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
  dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
           c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
  dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

  x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
  indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
  return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


# /////////////// End Distortions ///////////////


# /////////////// Further Setup ///////////////

def split_range(total, num_split, start_index=0):
  rs = np.linspace(start_index, total, num_split + 1).astype(np.int)
  result = [[rs[i], rs[i + 1]] for i in range(len(rs) - 1)]
  return result


def distort_iterate_and_save(method, severity, start_idx, end_idx):
  distorted_dataset = DistortImageFolder(
    root=args.validation_dir,
    method=method, severity=severity,
    start_idx=start_idx, end_idx=end_idx,
    transform='imagenet')
  # iterate to save distorted images
  for _ in distorted_dataset:
    continue
  
  return 0

  
def save_distorted(total_img_cnt, method=gaussian_noise, num_process=20):
  for severity in range(1, 6):
    print(method.__name__, severity)
    start = time.time()
    ranges = split_range(total_img_cnt, num_process)
    input_list = [(method, severity, idxs[0], idxs[1]) for idxs in ranges]
    
    pool = Pool(num_process)
    pool.starmap(distort_iterate_and_save, input_list)
    
    end = time.time()
    print('%f secs taken for %s %s' % (end-start, method.__name__, severity))


# /////////////// End Further Setup ///////////////


# /////////////// Display Results ///////////////
import collections

d = collections.OrderedDict()
d['Gaussian Noise'] = gaussian_noise
d['Shot Noise'] = shot_noise
d['Impulse Noise'] = impulse_noise
d['Defocus Blur'] = defocus_blur
d['Glass Blur'] = glass_blur
d['Motion Blur'] = motion_blur
d['Zoom Blur'] = zoom_blur
d['Snow'] = snow
d['Frost'] = frost
d['Fog'] = fog
d['Brightness'] = brightness
d['Contrast'] = contrast
d['Elastic'] = elastic_transform
d['Pixelate'] = pixelate
d['JPEG'] = jpeg_compression

d['Speckle Noise'] = speckle_noise
d['Gaussian Blur'] = gaussian_blur
d['Spatter'] = spatter
d['Saturate'] = saturate


# count total number of validation images first.
total_img_cnt = count_dataset(args.validation_dir)

print('\nTotal %d validation images. Distortion started.' % total_img_cnt)

# start distortion process
start = time.time()

for method_name in d.keys():
  save_distorted(total_img_cnt, d[method_name], num_process=args.num_workers)

end = time.time()

print('Total %f secs taken.' % (end-start))
