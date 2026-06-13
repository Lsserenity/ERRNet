from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import scipy.stats as st
import cv2
import numbers
import types
import collections
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import util.util as util
from scipy.signal import convolve2d


# utility
def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def arrshow(arr):
    Image.fromarray(arr.astype(np.uint8)).show()


def get_transform(opt):
    transform_list = []
    osizes = util.parse_args(opt.loadSize)
    fineSize = util.parse_args(opt.fineSize)
    if opt.resize_or_crop == 'resize_and_crop':    
        transform_list.append(
            transforms.RandomChoice([
                transforms.Resize([osize, osize], Image.BICUBIC) for osize in osizes
            ]))
        transform_list.append(transforms.RandomCrop(fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    return transforms.Compose(transform_list)


to_norm_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

to_tensor = transforms.ToTensor()


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


# functional 
def gaussian_blur(img, kernel_size, sigma):
    from scipy.ndimage.filters import gaussian_filter
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    img = np.asarray(img)
    # the 3rd dimension (i.e. inter-band) would be filtered which is unwanted for our purpose
    # new = gaussian_filter(img, sigma=sigma, truncate=truncate)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    elif isinstance(kernel_size, collections.Sequence):
        assert len(kernel_size) == 2        
    new = cv2.GaussianBlur(img, kernel_size, sigma)  # apply gaussian filter band by band    
    return Image.fromarray(new)


# transforms
class GaussianBlur(object):
    def __init__(self, kernel_size=11, sigma=3):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        return gaussian_blur(img, self.kernel_size, self.sigma)


class ReflectionSythesis_1(object):
    """Reflection image data synthesis for weakly-supervised learning 
    of ICCV 2017 paper *"A Generic Deep Architecture for Single Image Reflection Removal and Image Smoothing"*    
    """
    def __init__(self, kernel_sizes=None, low_sigma=2, high_sigma=5, low_gamma=1.3, high_gamma=1.3):
        self.kernel_sizes = kernel_sizes or [11]
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        self.low_gamma = low_gamma
        self.high_gamma = high_gamma
        print('[i] reflection sythesis model: {}'.format({
            'kernel_sizes': kernel_sizes, 'low_sigma': low_sigma, 'high_sigma': high_sigma,
            'low_gamma': low_gamma, 'high_gamma': high_gamma}))

    def __call__(self, B, R):
        if not _is_pil_image(B):
            raise TypeError('B should be PIL Image. Got {}'.format(type(B)))
        if not _is_pil_image(R):
            raise TypeError('R should be PIL Image. Got {}'.format(type(R)))
        
        B_ = np.asarray(B, np.float32) / 255.
        R_ = np.asarray(R, np.float32) / 255.

        kernel_size = np.random.choice(self.kernel_sizes)
        sigma = np.random.uniform(self.low_sigma, self.high_sigma)
        gamma = np.random.uniform(self.low_gamma, self.high_gamma)
        R_blur = R_
        kernel = cv2.getGaussianKernel(11, sigma)
        kernel2d = np.dot(kernel, kernel.T)

        for i in range(3):
            R_blur[...,i] = convolve2d(R_blur[...,i], kernel2d, mode='same')

        M_ = B_ + R_blur
        
        if np.max(M_) > 1:
            m = M_[M_ > 1]
            m = (np.mean(m) - 1) * gamma
            R_blur = np.clip(R_blur - m, 0, 1)
            M_ = np.clip(R_blur + B_, 0, 1)
        
        return B_, R_blur, M_


class ReflectionSythesisRealistic(object):
    """Realistic reflection synthesis with local masks and photometric jitter.

    The synthesized image follows M = clip(alpha * B + beta * A * R, 0, 1),
    where R is blurred, color-jittered, and slightly shifted, and A is a
    low-frequency local reflection mask.
    """
    def __init__(
            self,
            kernel_sizes=None,
            low_sigma=1.0,
            high_sigma=6.0,
            alpha_range=(0.85, 1.0),
            beta_range=(0.15, 0.6),
            shift_range=10,
            brightness_range=(0.8, 1.2),
            contrast_range=(0.8, 1.2),
            saturation_range=(0.8, 1.2),
            gamma_range=(0.8, 1.25),
            mask_blur_range=(21, 61)):
        self.kernel_sizes = kernel_sizes or [3, 5, 7, 9, 11, 15]
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.shift_range = shift_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.gamma_range = gamma_range
        self.mask_blur_range = mask_blur_range
        print('[i] realistic reflection sythesis model: {}'.format({
            'kernel_sizes': self.kernel_sizes, 'low_sigma': low_sigma, 'high_sigma': high_sigma,
            'alpha_range': alpha_range, 'beta_range': beta_range, 'shift_range': shift_range}))

    @staticmethod
    def _odd_kernel(value):
        value = int(value)
        if value < 1:
            value = 1
        if value % 2 == 0:
            value += 1
        return value

    @staticmethod
    def _sample_range(value_range):
        low, high = value_range
        return np.random.uniform(low, high)

    def _jitter_reflection(self, R):
        img = Image.fromarray(np.uint8(np.clip(R, 0, 1) * 255.0))
        img = ImageEnhance.Brightness(img).enhance(self._sample_range(self.brightness_range))
        img = ImageEnhance.Contrast(img).enhance(self._sample_range(self.contrast_range))
        img = ImageEnhance.Color(img).enhance(self._sample_range(self.saturation_range))
        R = np.asarray(img, np.float32) / 255.0

        gamma = self._sample_range(self.gamma_range)
        R = np.power(np.clip(R, 0, 1), gamma)
        return R

    def _shift_reflection(self, R):
        if self.shift_range <= 0:
            return R

        h, w = R.shape[:2]
        dx = np.random.randint(-self.shift_range, self.shift_range + 1)
        dy = np.random.randint(-self.shift_range, self.shift_range + 1)
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(R, matrix, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    def _local_mask(self, height, width):
        small_h = max(4, height // 16)
        small_w = max(4, width // 16)
        mask = np.random.rand(small_h, small_w).astype(np.float32)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_CUBIC)

        low, high = self.mask_blur_range
        kernel_size = self._odd_kernel(np.random.randint(low, high + 1))
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        mask = mask - mask.min()
        mask = mask / (mask.max() + 1e-6)
        return mask[..., None].astype(np.float32)

    def __call__(self, B, R):
        if not _is_pil_image(B):
            raise TypeError('B should be PIL Image. Got {}'.format(type(B)))
        if not _is_pil_image(R):
            raise TypeError('R should be PIL Image. Got {}'.format(type(R)))

        B_ = np.asarray(B, np.float32) / 255.0
        R_ = np.asarray(R, np.float32) / 255.0

        R_ = self._jitter_reflection(R_)
        R_ = self._shift_reflection(R_)

        kernel_size = self._odd_kernel(np.random.choice(self.kernel_sizes))
        sigma = np.random.uniform(self.low_sigma, self.high_sigma)
        R_blur = cv2.GaussianBlur(R_, (kernel_size, kernel_size), sigma)

        mask = self._local_mask(B_.shape[0], B_.shape[1])
        alpha = self._sample_range(self.alpha_range)
        beta = self._sample_range(self.beta_range)

        R_masked = np.clip(mask * R_blur, 0, 1)
        M_ = np.clip(alpha * B_ + beta * R_masked, 0, 1)

        return np.float32(B_), np.float32(R_masked), np.float32(M_)


class Sobel(object):
    def __call__(self, img):
        if not _is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        gray_img = np.array(img.convert('L'))
        x = cv2.Sobel(gray_img,cv2.CV_16S,1,0)
        y = cv2.Sobel(gray_img,cv2.CV_16S,0,1)
        
        absX = cv2.convertScaleAbs(x)   
        absY = cv2.convertScaleAbs(y)
        
        dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
        return Image.fromarray(dst)


class ReflectionSythesis_2(object):
    """Reflection image data synthesis for weakly-supervised learning 
    of CVPR 2018 paper *"Single Image Reflection Separation with Perceptual Losses"*
    """
    def __init__(self, kernel_sizes=None):
        self.kernel_sizes = kernel_sizes or np.linspace(1,5,80)
    
    @staticmethod
    def gkern(kernlen=100, nsig=1):
        """Returns a 2D Gaussian kernel array."""
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        kernel = kernel/kernel.max()
        return kernel

    def __call__(self, t, r):        
        t = np.float32(t) / 255.
        r = np.float32(r) / 255.
        ori_t = t
        # create a vignetting mask
        g_mask=self.gkern(560,3)
        g_mask=np.dstack((g_mask,g_mask,g_mask))
        sigma=self.kernel_sizes[np.random.randint(0, len(self.kernel_sizes))]

        t=np.power(t,2.2)
        r=np.power(r,2.2)
        
        sz=int(2*np.ceil(2*sigma)+1)
        
        r_blur=cv2.GaussianBlur(r,(sz,sz),sigma,sigma,0)
        blend=r_blur+t
        
        att=1.08+np.random.random()/10.0
        
        for i in range(3):
            maski=blend[:,:,i]>1
            mean_i=max(1.,np.sum(blend[:,:,i]*maski)/(maski.sum()+1e-6))
            r_blur[:,:,i]=r_blur[:,:,i]-(mean_i-1)*att
        r_blur[r_blur>=1]=1
        r_blur[r_blur<=0]=0

        h,w=r_blur.shape[0:2]
        neww=np.random.randint(0, 560-w-10)
        newh=np.random.randint(0, 560-h-10)
        alpha1=g_mask[newh:newh+h,neww:neww+w,:]
        alpha2 = 1-np.random.random()/5.0
        r_blur_mask=np.multiply(r_blur,alpha1)
        blend=r_blur_mask+t*alpha2
        
        t=np.power(t,1/2.2)
        r_blur_mask=np.power(r_blur_mask,1/2.2)
        blend=np.power(blend,1/2.2)
        blend[blend>=1]=1
        blend[blend<=0]=0
        
        return np.float32(ori_t), np.float32(r_blur_mask), np.float32(blend)


# Examples
if __name__ == '__main__':
    """cv2 imread"""
    # img = cv2.imread('testdata_reflection_real/19-input.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img2 = cv2.GaussianBlur(img, (11,11), 3)    

    """Sobel Operator"""
    # img = np.array(Image.open('datasets/VOC224/train/B/2007_000250.png').convert('L'))


    """Reflection Sythesis"""
    b = Image.open('datasets/VOCsmall/train/B/2008_000148.png')
    r = Image.open('datasets/VOCsmall/train/B/2007_000243.png')
    G = ReflectionSythesis_1()
    m, r = G(b, r)
    r.show()
    
    # img2 = gaussian_blur(img, 11, 3)
    # img2 = GaussianBlur(1, 1)(img)
    # print(np.sum(np.array(img2) - np.array(img)))
    # img2.show()
