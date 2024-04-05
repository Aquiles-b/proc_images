from noise import sp_noise
import cv2
import sys
import time
import numpy as np


# Faz o stacking de imagens ruidosas usando a média dos pixels até que a
# diferença de PSNR entre duas iterações seja menor que @stop_distance 
def stacking_test(img_original, noise_level: float, stop_distance: float) -> float:
    previous_psnr = 0
    psnr = 0
    img_stack = []
    while(True):
        img_stack.append(sp_noise(img_original, noise_level))
        img = np.mean(img_stack, axis=0).astype(np.uint8)
        psnr = cv2.PSNR(img_original, img)
        if psnr - previous_psnr < stop_distance:
            return psnr
        previous_psnr = psnr

def main() -> None:
    noise_levels = [0.01, 0.02, 0.05, 0.07, 0.1]
    img_original = cv2.imread('original.jpg')
    psnr_ceil = cv2.PSNR(img_original, img_original)

    # cvBlur, cvGaussianBlur, cvMedianBlur e Stacking

    filters = { 'cv_blur': [], 'cv_ga_blur': [], 
               'cv_me_blur': [], 'stacking': [] }

    ks = 3
    stacking_test(img_original, noise_levels[4], 0.05)
    # for noise_level in noise_levels:
    #     img_noised = sp_noise(img_original, noise_level)
    #     psnr_cv_blur = cv2.PSNR(img_original, cv2.blur(img_noised, (ks, ks)))
    #     psnr_cv_ga_blur = cv2.PSNR(img_original, cv2.GaussianBlur(img_noised, (ks, ks), 0))
    #     psnr_cv_me_blur = cv2.PSNR(img_original, cv2.medianBlur(img_noised, ks))



if __name__ == '__main__':
    main()
