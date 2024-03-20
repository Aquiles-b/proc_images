import cv2
import sys


def main() -> None:
    img_in_name = sys.argv[1]
    img_out_name = sys.argv[2]

    img = cv2.imread(img_in_name, cv2.COLOR_BGR2HSV)

    cv2.imshow('Teste', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
