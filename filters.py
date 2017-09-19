#!/usr/bin/env python
import matplotlib.image as mpimg
import cv2
import numpy as np
import math


def soebel_filter(img, direction='x', sobel_kernel=7, thresh_min=37, thresh_max=169):
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the derivative in the  direction
    if direction == 'x':
        _sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif direction == 'y':
        _sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(_sobel)
    # Convert the absolute value image to 8-bit:
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    sxbinary = np.zeros_like(scaled_sobel, np.uint8)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return sxbinary

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=7, thresh_min=85, thresh_max=168):
    # Convert to grayscale
    gray = cv2.cvtColor(np.copy(img), cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag, np.uint8)
    binary_output[(gradmag >= thresh_min) & (gradmag <= thresh_max)] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_thresh(img, sobel_kernel=7, thresh_min=0, thresh_max=np.pi/2):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir, np.uint8)
    binary_output[(absgraddir >= thresh_min) & (absgraddir <= thresh_max)] = 1

    # Return the binary image
    return binary_output

# Define a function that thresholds the S-channel of HLS
def s_select(img, thresh_min=162, thresh_max=255):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel, np.uint8)
    binary_output[(s_channel > thresh_min) & (s_channel <= thresh_max)] = 1
    
    return binary_output

# Define a function that thresholds the S-channel of HLS
def l_select(img, thresh_min=123, thresh_max=141):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,1]
    binary_output = np.zeros_like(s_channel, np.uint8)
    binary_output[(s_channel > thresh_min) & (s_channel <= thresh_max)] = 1
    
    return binary_output

def h_select(img, thresh_min=20, thresh_max=25):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,0]
    binary_output = np.zeros_like(s_channel, np.uint8)
    binary_output[(s_channel > thresh_min) & (s_channel <= thresh_max)] = 1
    
    return binary_output

def laplacian(img, thresh_min=0, thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    abs_laplacian = np.absolute(_laplacian)
    scaled_laplacian = np.uint8(255*abs_laplacian/np.max(abs_laplacian))

    sxbinary = np.zeros_like(scaled_laplacian, np.uint8)
    sxbinary[(scaled_laplacian >= thresh_min) & (scaled_laplacian <= thresh_max)] = 1
    
    return sxbinary

def scharr_diagonal(img, thresh_min=0, thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = np.array([[10, 3, 0],[3, 0, -3],[0, -3, -10]]) 
    scharr_diagonal = cv2.filter2D(gray, -1, kernel)

    sxbinary = np.zeros_like(scharr_diagonal, np.uint8)
    sxbinary[(scharr_diagonal >= thresh_min) & (scharr_diagonal <= thresh_max)] = 1
    
    return sxbinary