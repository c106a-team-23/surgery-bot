#!/usr/bin/env python
"""Segmentation skeleton code for Lab 6
Course: EECS C106A, Fall 2019
Author: Grant Wang

This Python file is the skeleton code for Lab 3. You are expected to fill in
the body of the incomplete functions below to complete the lab. The 'test_..'
functions are already defined for you for allowing you to check your 
implementations.

When you believe you have completed implementations of all the incompeleted
functions, you can test your code by running python segmentation.py at the
command line and step through test images
"""

import os

from PIL.Image import NONE
import numpy as np
import sys as sys
import cv2
import matplotlib.pyplot as plt

this_file = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = '/'.join(this_file.split('/')[:-2]) + '/img'

def read_image(img_name, grayscale=False):
    """ reads an image

    Parameters
    ----------
    img_name : str
        name of image
    grayscale : boolean
        true if image is in grayscale, false o/w
    
    Returns
    -------
    ndarray
        an array representing the image read (w/ extension)
    """

    if not grayscale:
        img = cv2.imread(img_name)
    else:
        img = cv2.imread(img_name, 0)

    return img

def write_image(img, img_name):
    """writes the image as a file
    
    Parameters
    ----------
    img : ndarray
        an array representing an image
    img_name : str
        name of file to write as (make sure to put extension)
    """

    cv2.imwrite(img_name, img)

def show_image(img_name, title='Fig', grayscale=False):
    """show the  as a matplotlib figure
    
    Parameters
    ----------
    img_name : str
        name of image
    tile : str
        title to give the figure shown
    grayscale : boolean
        true if image is in grayscale, false o/w
    """

    if not grayscale:
        plt.imshow(img_name)
        plt.title(title)
        plt.show()
    else:
        plt.imshow(img_name, cmap='gray')
        plt.title(title)
        plt.show()


def threshold_segment_naive(gray_img, lower_thresh, upper_thresh):
    """perform grayscale thresholding using a lower and upper threshold by
    blacking the background lying between the threholds and whitening the
    foreground

    Parameter
    ---------
    gray_img : ndarray
        grayscale image array
    lower_thresh : float or int
        lowerbound to threshold (an intensity value between 0-255)
    upper_thresh : float or int
        upperbound to threshold (an intensity value between 0-255)

    Returns
    -------
    ndarray
        thresholded version of gray_img
    """
    # TODO: Implement threshold segmentation by setting pixels of gray_img inside the 
    # lower_thresh and upper_thresh parameters to 0
    # Then set any value that is outside the range to be 1 
    # Hints: make a copy of gray_img so that we don't alter the original image
    # Boolean array indexing, or masking will come in handy. 
    # See https://docs.scipy.org/doc/numpy-1.13.0/user/basics.indexing.html
    gray_img_mod = np.copy(gray_img)
    within_thresh = (gray_img_mod>=lower_thresh).astype(int) + (gray_img_mod<=upper_thresh).astype(int)
    return within_thresh

    # np.set_printoptions(threshold=sys.maxsize)
    # print(gray_img_mod)

    # raise NotImplementedError()

def edge_detect_naive(gray_img):
    """perform edge detection using first two steps of Canny (Gaussian blurring and Sobel
    filtering)

    Parameter
    ---------
    gray_img : ndarray
        grayscale image array

    Returns
    -------
    ndarray
        gray_img with edges outlined
    """

    gray_s = gray_img.astype('int16') # convert to int16 for better img quality 
    # TODO: Blur gray_s using Gaussian blurring, convole the blurred image with
    # Sobel filters, and combine to compute the intensity gradient image (image with edges highlighted)
    # Hints: open-cv GaussianBlur will be helpful https://medium.com/analytics-vidhya/gaussian-blurring-with-python-and-opencv-ba8429eb879b 
    # Use opencv's filter2D to perform the convolution.

    # Steps
    # 1. apply a gaussian blur with a 5x5 kernel.
    # 2. define the convolution kernel Kx and Ky as defined in the doc.
    # 3. compute Gx and Gy by convolving Kx and Ky respectively with the blurred image.
    # 4. compute G = sqrt(Gx ** 2 + Gy ** 2)
    # 5. Return G

    # Comment out this line after you implement the function.
    # raise NotImplementedError()

    # First, apply a gaussian blur with a 5x5 kernel.
    blur =  cv2.GaussianBlur(gray_s, (5, 5), 0)
    # Next, define the convolution kernel Kx and Ky as defined in the doc.
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    # Now, compute Gx and Gy by convolving Kx and Ky respectively with the blurred image.

    G_x = cv2.filter2D(blur, -1, Kx)
    G_y = cv2.filter2D(blur, -1, Ky)
    
    # Finally, compute G = sqrt(Gx ** 2 + Gy ** 2)
    G = np.sqrt(G_x**2 + G_y**2)
    # np.set_printoptions(threshold=sys.maxsize)
    # print(G)
    # Return G
    return G


def edge_detect_canny(gray_img):
    """perform Canny edge detection

    Parameter
    ---------
    gray_img : ndarray
        grayscale image array

    Returns
    -------
    ndarray
        gray_img with edges outlined
    """

    edges = cv2.Canny(gray_img, 100, 200)

    return edges

def do_kmeans(data, n_clusters):
    """Uses opencv to perform k-means clustering on the data given. Clusters it into
       n_clusters clusters.

       Args:
         data: ndarray of shape (n_datapoints, dim)
         n_clusters: int, number of clusters to divide into.

       Returns:
         clusters: integer array of length n_datapoints. clusters[i] is
         a number in range(n_clusters) specifying which cluster data[i]
         was assigned to. 
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, clusters, centers = kmeans = cv2.kmeans(data.astype(np.float32), n_clusters, bestLabels=None, criteria=criteria, attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)

    return clusters

def cluster_segment(img, n_clusters, random_state=0):
    """segment image using k_means clustering

    Parameter
    ---------
    img : ndarray
        rgb image array
    n_clusters : int
        the number of clusters to form as well as the number of centroids to generate
    random_state : int
        determines random number generation for centroid initialization

    Returns
    -------
    ndarray
        clusters of gray_img represented with similar pixel values
    """
    # Remove this line when you implement this function.
    # raise NotImplementedError()

    # Downsample img by a factor of 2 first using the mean to speed up K-means
    img_d = cv2.resize(img, dsize=(img.shape[1]/2, img.shape[0]/2), interpolation=cv2.INTER_NEAREST)

    img_d = cv2.GaussianBlur(img_d, (5, 5), 0)
    # TODO: Generate a clustered image using K-means

    # first convert our 3-dimensional img_d array to a 2-dimensional array
    # whose shape will be (length * width, number of channels) hint: use img_d.shape
    # print(img_d.shape)
    img_r = img_d.reshape((img_d.shape[0]*img_d.shape[1], 3))
    
    # fit the k-means algorithm on this reshaped array img_r using the
    # the do_kmeans function defined above.
    clusters = do_kmeans(img_r, n_clusters)

    # reshape this clustered image to the original downsampled image (img_d) shape
    cluster_img = np.reshape(clusters, (img_d.shape[0], img_d.shape[1]))

    # Upsample the image back to the original image (img) using nearest interpolation
    img_u = cv2.resize(src=cluster_img, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return img_u.astype(np.uint8)


def to_grayscale(rgb_img):
    return np.dot(rgb_img[... , :3] , [0.299 , 0.587, 0.114])

def segment_image(img, n_clusters = 3): 
    # ONLY USE ONE SEGMENTATION METHOD

    # perform thresholding segmentation
    # binary = threshold_segment_naive(to_grayscale(img), 150, 200).astype(np.uint8)
    
    # perform clustering segmentation.  
    # binary = cluster_segment(img, n_clusters).astype(np.uint8)

    # if np.mean(binary) > 0.5:
        # binary = 1 - binary #invert the pixels if K-Means assigned 1's to background, and 0's to foreground
    # sections = discretize(img, n=10)
    # for i, s in enumerate(sections):
    #     show_image(s, title="section{}".format(i))
        
    #     clusters = (cluster_segment(s, n_clusters) * (255 / (n_clusters-1))).astype(np.uint8)
    #     cv2.imwrite(IMG_DIR + "/cluster.jpg", clusters)
    #     clusters = cv2.imread(IMG_DIR + '/cluster.jpg')  
    
    binary = cluster_segment(img, n_clusters).astype(np.uint8)

    if binary[binary.shape[0]-1, binary.shape[1]-1] > 0:
        binary = 1 - binary 
    # show_image(binary, "clustered")
    
    return binary


"""
below are tests used for sanity checking you image, edit as you see appropriate

"""

def test_thresh_naive(img, lower_thresh, upper_thresh):
    thresh = threshold_segment_naive(img, lower_thresh, upper_thresh)
    show_image(thresh, title='thresh naive', grayscale=True)
    cv2.imwrite(IMG_DIR + "/thresh.jpg", thresh.astype('uint8') * 255)

def test_edge_naive(img):
    edges = edge_detect_naive(img)
    show_image(edges, title='edge naive', grayscale=True)
    cv2.imwrite(IMG_DIR + "/edges.jpg", edges)

def test_edge_canny(img):
    edges = edge_detect_canny(img)
    show_image(edges, title='edge canny', grayscale=True)
    cv2.imwrite(IMG_DIR + "/edges_canny.jpg", edges)

def test_cluster(img, n_clusters):
    # For visualization, we need to scale up the image so it
    # is in range(256) instead of range(n_clusters).
    # object_detector = cv2.createBackgroundSubtractorMOG2()
    # mask = object_detector.apply(img)
    # show_image(mask, title='mask')

    # clusters = (cluster_segment(img, n_clusters) * (255 / (n_clusters-1))).astype(np.uint8)
    # cv2.imwrite(IMG_DIR + "/cluster.jpg", clusters)
    # clusters = cv2.imread(IMG_DIR + '/cluster.jpg')
    # show_image(clusters, title='cluster')
    sections = discretize(img, n=5)
    for i, s in enumerate(sections):
        show_image(s, title="section{}".format(i))
        
        clusters = (cluster_segment(s, n_clusters) * (255 / (n_clusters-1))).astype(np.uint8)
        cv2.imwrite(IMG_DIR + "/cluster.jpg", clusters)
        clusters = cv2.imread(IMG_DIR + '/cluster.jpg')
        show_image(clusters, title='cluster')

    #focusing on a specific region

def check_for_close():
    # prediction = model.predict(data)
    # if predicion == 0:
    #     return False
    # return True
    return False

def discretize(image, n= 4):
    # print(image.shape)
    # print(image[0])
    h, w, _  = image.shape
    size = h//n
    sections = [(((i-1)*size, i*size), image[(i-1)*size:i*size, : , : ]) for i in range(1 , n+1)]
    # sections = np.array(sections)
    return sections
    
if __name__ == '__main__':
    # adjust the file names here
    # test_img = read_image(IMG_DIR + '/legos.jpg', grayscale=True)
    # test_img_color = read_image(IMG_DIR + '/legos.jpg')

    # # uncomment the test you want to run
    # # it will plot the image and also save it

    # test_thresh_naive(test_img, 0, 230)
    # test_edge_naive(test_img)
    # test_edge_canny(test_img)
    # test_cluster(test_img_color, 5)

    # test_img = read_image(IMG_DIR + '/lego.jpg', grayscale=True)
    # test_img_color = read_image(IMG_DIR + '/lego.jpg')

    # test_thresh_naive(test_img, 90, 185)
    # test_edge_naive(test_img)
    # test_edge_canny(test_img)
    # test_cluster(test_img_color, 2)
    
    test_img = read_image(IMG_DIR + '/staples.jpg', grayscale=True)
    test_img_color = read_image(IMG_DIR + '/staples.jpg')
    # test_thresh_naive(test_img, 90, 185)
    # test_edge_naive(test_img)
    # test_edge_canny(test_img)
    test_cluster(test_img_color, 2)

    # from keras.models import load_model
    # from PIL import Image, ImageOps

    # model = load_model('path')
    # data = np.darray(shape=(1, 224, 224, 3), dtype=np.float32)
    # image = Image.open('path')
    # image = Image.Ops.fit(image, (224, 224), Image.ANTIALIAS)
    # image_array = np.asarray(image)
    # normalized_image_array = (image_array.astype(np.float32)/127.0) - 1
    # data[0] = normlized_image_array
    # prediction = model.predict(data)
    

