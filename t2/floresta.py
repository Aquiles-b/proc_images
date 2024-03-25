import cv2
import sys
import numpy as np


def main() -> None:
    img_in_name = sys.argv[1]
    img_out_name = sys.argv[2]

    img = cv2.imread(img_in_name)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (0, 0, 0), (75, 255, 255))

    imask = mask>0
    orange=np.zeros_like(img,np.uint8)
    orange[imask]=img[imask]
    yellow=img.copy()
    hsv[...,0] = hsv[...,0]+20
    yellow[imask]=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)[imask]
    yellow=np.clip(yellow,0,255)

    cv2.imshow("img",yellow)
    cv2.waitKey()
    nofish=img.copy()
    nofish=cv2.bitwise_and(nofish,nofish,mask=(np.bitwise_not(imask)).astype(np.uint8))
    cv2.imshow("img",nofish)
    cv2.waitKey()



if __name__ == "__main__":
    main()
