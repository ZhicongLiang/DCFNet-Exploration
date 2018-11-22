'''
Ref: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
image : ndarray
Input image data. Will be converted to float.
mode : str
One of the following strings, selecting the type of noise to add:

'gauss'     Gaussian-distributed additive noise.
'poisson'   Poisson-distributed noise generated from the data.
's&p'       Replaces random pixels with 0 or 1.
'speckle'   Multiplicative noise using out = image + n*image,where
            n is uniform noise with specified mean & variance.
'''
import torch
import matplotlib.pyplot as plt
import numpy as np

def gauss_noise(var=0.01):
    def fun(image):
        ch, row, col = image.shape
        mean = 0
        sigma = var**0.5
        normal = torch.distributions.normal.Normal(mean, sigma)
        gauss = normal.sample((ch, row, col))
        noisy = image + gauss
        return noisy
    return fun

def salt_pepper_noise(amount=0.004):
    def fun(image):
        ch, row,col = image.shape
        s_vs_p = 0.5
        out = torch.clone(image)
        # Salt mode
        num_salt = np.ceil(amount * np.prod(image.shape) * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1
    
        # Pepper mode
        num_pepper = np.ceil(amount* np.prod(image.shape) * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    return fun

def identity(image):
    return image