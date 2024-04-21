import sys
from glob import glob
import cv2
import os
from matplotlib import pyplot as plt

# Retorna um dicionário com os dados dos formulários.
def catch_forms_data(forms: list) -> list:
    for form in forms[:1]:
        pass

def catch_regions(forms: list) -> list:
    img = cv2.imread(forms[0], cv2.IMREAD_GRAYSCALE)
    q1_i = 670
    q2_i = 840
    q3_i = 1040
    q4_i = 1200
    q5_i = 1400

    img_q1 = img[q1_i:q1_i+200]
    img_q2 = img[q2_i:q2_i+200]
    img_q3 = img[q3_i:q3_i+200]
    img_q4 = img[q4_i:q4_i+200]
    img_q5 = img[q5_i:q5_i+200]


def main() -> None:
    argc = len(sys.argv)
    if argc == 1:
        print("Usage: python3 forms.py <forms-dir> <forms-out-dir>")
        sys.exit(1)

    forms_dir = sys.argv[1]
    if not os.path.exists(forms_dir):
        print(f"Error: {forms_dir} does not exist.")
        sys.exit(1)

    forms = sorted(glob(f"{forms_dir}/*.png"))
    # catch_forms_data(forms)
    catch_regions(forms)

if __name__ == '__main__':
    main()
