import cv2
import sys
import numpy as np


def main() -> None:
    img_in_name = sys.argv[1]
    img_out_name = sys.argv[2]

    img = cv2.imread(img_in_name)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    low_forest = np.array([30, 0, 0])
    high_forest = np.array([106, 255, 255])

    # Pega a mascara da floresta por range de cor
    mask_forest = cv2.inRange(hsv, low_forest, high_forest)

    b, g, r = cv2.split(img)
    # Pega a mascara do rio por canais
    _, mask_river_b = cv2.threshold(b, 130, 255, cv2.THRESH_BINARY)
    _, mask_river_r = cv2.threshold(r, 170, 255, cv2.THRESH_BINARY)

    # Pega a mascara da floresta do canal g
    _, mask_forest_g = cv2.threshold(g, 110, 255, cv2.THRESH_BINARY)

    # soma as mascaras dos rios
    mask_river = cv2.bitwise_or(mask_river_b, mask_river_r)

    # subtrai a mascara do rio da mascara da floresta
    mask_forest = cv2.bitwise_and(mask_forest, cv2.bitwise_not(mask_river))

    # subtrai a mascara b do rio da mascara g da floresta
    mask_forest_g = cv2.bitwise_and(mask_forest_g, cv2.bitwise_not(mask_river_b))

    # Soma a mascara do canal g com a mascara da floresta
    mask_forest = cv2.bitwise_or(mask_forest, mask_forest_g)

    mask_forest = cv2.bitwise_not(mask_forest)

    img_seg = img.copy()
    img_seg = cv2.bitwise_and(img_seg, img_seg, mask=mask_forest)

    cv2.imshow("original", img)
    cv2.imshow("mask", img_seg)
    cv2.moveWindow("original", 0, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
