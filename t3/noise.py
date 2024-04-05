import sys
import numpy as np
import cv2 as cv
import random

###------------------------------------
### Add Noise with a give probability
###------------------------------------
def sp_noise(image,prob):
        
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output


def main(argv):

    if (len(sys.argv)!= 3):
        sys.exit("Use: stack <imageIn> <imageOut>>")

    # ler a imagem
    img = cv.imread(argv[1], 0)
    imgNoise = sp_noise (img, 0.05)



    psnr = cv.PSNR(img, imgNoise)
    print ('PSNR = ', psnr)
    
    psnr = cv.PSNR(img, img)
    print ('PSNR Max = ', psnr)

    cv.imwrite("noise.png", imgNoise)


if __name__ == '__main__':
    main(sys.argv)


