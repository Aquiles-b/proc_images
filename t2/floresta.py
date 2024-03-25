import cv2
import sys
import numpy as np


def main() -> None:
    img_in_name = sys.argv[1]
    img_out_name = sys.argv[2]

    img = cv2.imread(img_in_name)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    low = np.array([0 ,0 ,0])
    high = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, low, high)
    mask = cv2.bitwise_not(mask)

    img_seg = img.copy()
    img_seg = cv2.bitwise_and(img_seg, img_seg, mask=mask)

    cv2.imshow("original", img)
    cv2.imshow("mask", img_seg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
