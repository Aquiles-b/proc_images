import sys
from glob import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# Retorna um dicionário com os dados dos formulários.
def catch_forms_data(forms: list) -> list:
    for form in forms[:1]:
        pass

def catch_regions(form: np.ndarray):
    img = cv2.imread(form, cv2.IMREAD_GRAYSCALE)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    q_w = 850
    # Inicio de cada questão
    q1_i = 670
    q2_i = 840
    q3_i = 1040
    q4_i = 1200
    q5_i = 1400
    q6_i = 1580
    # Inicio das questões de sim ou não
    q7_i = 1770
    q8_i = 1970
    # Inicio da questão de nota
    rating_i = 2270
    # Recortando as regiões
    img_q1 = img[q1_i:q1_i+200, q_w:]
    img_q2 = img[q2_i:q2_i+200, q_w:]
    img_q3 = img[q3_i:q3_i+200, q_w:]
    img_q4 = img[q4_i:q4_i+200, q_w:]
    img_q5 = img[q5_i:q5_i+200, q_w:]
    img_q6 = img[q6_i:q6_i+200, q_w:]

    img_q7 = img[q7_i:q7_i+200, 0:1600]
    img_q8 = img[q8_i:q8_i+300]
    img_rating = img[rating_i:rating_i+200, 700:]

    print(f'\n{form}:')
    analyze_question_type_1(img_q1)
    analyze_question_type_1(img_q2)
    analyze_question_type_1(img_q3)
    analyze_question_type_1(img_q4)
    analyze_question_type_1(img_q5)
    analyze_question_type_1(img_q6)

def analyze_question_type_1(img: np.ndarray) -> int:
    offset = 445
    img_exc = img[:, 0:offset]
    img_good = img[:, offset:offset*2]
    img_fair = img[:, offset*2:offset*3]
    img_poor = img[:, offset*3:]

    # count the number of black pixels
    exc_black = np.sum(img_exc == 0)
    good_black = np.sum(img_good == 0)
    fair_black = np.sum(img_fair == 0)
    poor_black = np.sum(img_poor == 0)

    print(exc_black, good_black, fair_black, poor_black)


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
    for form in forms[:3]:
        catch_regions(form)

if __name__ == '__main__':
    main()
