# -*- coding: utf-8 -*-
"""
Created on Mon May 17 21:54:17 2021

@author: andre
"""
### IMPORTACIÓ DE LES EINES 
import cv2
import numpy as np
from skimage.morphology import skeletonize
import fingerprint_enhancer  
import os
import random

##### CARGUEM LES DADES DEL TEST I DEL TRAINING (apliquem al 50% del test i al 50% del training soroll)

def applyNoiseSaltAndPepper(image,prob):    ################ REVISAR FUNCIONAMENT
    #Llegim la imatge
    #image = cv2.imread(name_img, cv2.IMREAD_GRAYSCALE)
    output = np.zeros(image.shape,np.uint8)
    threshold = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            elem = random.random()
            if elem < prob:
                output[i][j] = 0
            elif elem > threshold:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def applyNoiseBlurring(imatge):    
    #Llegim la imatge
    Gaussian = cv2.GaussianBlur(imatge, (7, 7), 0)
    return Gaussian


def applyNoiseGaussian(imatge):    

    # Generate Gaussian noise
    gauss = np.random.normal(0,1,imatge.size)
    gauss = gauss.reshape(imatge.shape[0],imatge.shape[1]).astype('uint8')
    
    # Add the Gaussian noise to the image
    img_gauss = cv2.add(imatge,gauss)
    
    # Display the image
    #cv2.imshow('a',img_gauss)
    #cv2.waitKey(0)

    return img_gauss

img_test = []
img_training = []

for file_train in os.listdir("C:/Users/andre/Desktop/PROYECTO PSIV/img1-N-training/"):
    if file_train.split(".")[-1].lower() == "png":
        img = cv2.imread("C:/Users/andre/Desktop/PROYECTO PSIV/img1-N-training/" + file_train, cv2.IMREAD_GRAYSCALE)
        img_training.append(img)
        
for file_test in os.listdir("C:/Users/andre/Desktop/PROYECTO PSIV/img1-N-test/"):
    if file_test.split(".")[-1].lower() == "png":
        img = cv2.imread("C:/Users/andre/Desktop/PROYECTO PSIV/img1-N-test/" + file_test, cv2.IMREAD_GRAYSCALE)
        img_test.append(img)

### APLIQUEM SOROLL A LES IMATGES
#for i in range(0,20):
    #img_test[i] = applyNoiseSaltAndPepper(img_test[i],0.05)
    #img_training[i] = applyNoiseSaltAndPepper(img_training[i],0.05)

#####Exemple imatge amb soroll aplicat
#cv2.imshow('img',img_test[0])
#cv2.waitKey(0)

#### FUNCIONS NECESSARIES PER FER EL PROGRAMA 1-N  
def treure_pixels_aillats(imatge):
    
    connectivity = 8

    output = cv2.connectedComponentsWithStats(imatge, connectivity, cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]

    new_image = imatge.copy()

    for label in range(num_stats):
        if stats[label,cv2.CC_STAT_AREA] == 1:
            new_image[labels == label] = 0

    return new_image

def preprocessament(img):
    
    #Traiem soroll a les imatges
    img = treure_pixels_aillats(img)
    
    #Apliquem un filtre
    kernel_filter = np.ones((6,6),np.float32)/(6*6)
    filtered_img = cv2.filter2D(img,-1,kernel_filter)  
    
    #Fa una millora de la imatge
    img = fingerprint_enhancer.enhance_Fingerprint(filtered_img)
    img = np.array(img, dtype=np.uint8)
    
    #Aplica un threshold
    th, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    #Normalitza els valors de la imatge a 0 i a 1
    img[img == 255] = 1
    
    #Apliquem thinning a la imatge
    skeleton = skeletonize(img)
    skeleton = 255 - 255 * np.array(skeleton, dtype=np.uint8)
    
    return skeleton

def obtencio_minucies(img):
    
    #Apliquem el detector de Harris
    harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
    harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    
    threshold_harris = 125.0
    
    #Extraiem els keypoints
    keypoints = []
    for x in range(0, harris_normalized.shape[0]):
        for y in range(0, harris_normalized.shape[1]):
            
            if harris_normalized[x][y] > threshold_harris:
                keypoints.append(cv2.KeyPoint(y, x, 1))
                
    #Definim el objecte que calcularà els descriptors
    orb = cv2.ORB_create()
    
    #Càlcul dels descriptors
    _, descriptors = orb.compute(img, keypoints)
    
    return (keypoints, descriptors)


def matching_empremtes(img1, img2):
    
    img1 = preprocessament(img1)
    img2 = preprocessament(img2)
    
    kp1, des1 = obtencio_minucies(img1)
    kp2, des2 = obtencio_minucies(img2)
    
    #Definim el objecte per fer el match dels descriptors
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1, des2, None)
    
    #Calculem les distàncies dels descriptors i les ordenem 
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    #Només agafem el 15% dels millors matches
    numGoodMatches = int(len(matches) * 0.15)
    matches = matches[:numGoodMatches]
    
    #Obtenim els punts dels millors matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    score = 0
    
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
        score += match.distance
        
    ####Dibuixem els matches seleccionats
    #img_matching = cv2.drawMatches(img1, kp1, img2, kp2, matches, flags=2, outImg=None)
    #plt.imshow(img_matching)
    #plt.show()
    
    score_threshold = 33
    if score/len(matches) < score_threshold:
        return True, score/len(matches)
    else:
        return False, score/len(matches)
      

##### PROGRAMA 1-N
def programa1N(img, conjunt_img):
    llista=[]
    for imatge in conjunt_img:
        matching,score = matching_empremtes(img,imatge)
        if matching:
            llista.append([score,imatge])
    if llista!=[]:
        llista.sort(key=lambda x: float(x[0]))
        percentatge_matching = 100-llista[0][0]
        
        print("Matching! El percentatge de matching és el següent:",percentatge_matching)
        cv2.imshow('Imatge amb la que fa matching',llista[0][1])
        cv2.waitKey(0)
            
    else:     
        print("L'empremta no és troba en el conjunt de dades :(")
    
img = cv2.imread("C:/Users/andre/Desktop/PROYECTO PSIV/6_img_escollides/im1_2.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow('imatge seleccionada', img)
cv2.waitKey(0)

img = preprocessament(img)
import time
start_time = time.time()
programa1N(img,img_training)
print("--- %s seconds ---" % (time.time() - start_time))

