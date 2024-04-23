import sys
from glob import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


# Retorna um dicionário com os dados dos formulários.
def catch_forms_data(forms: list) -> list:
    for form in forms:
        img = cv2.imread(form, cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        form_type = catch_form_type(img)
        regions = catch_regions(img)

        if form_type == 0:
            print(f"Form type: {form_type}, {form}")
            analyze_form_0(regions)

# Retorna o tipo do formulário.
def catch_form_type(form: np.ndarray) -> int:
    # (912, 17, 606, 156)
    img_guess_f = form[16:175, 900:1520]
    bp = np.sum(img_guess_f == 0)

    if bp > 15000:
        return 0
    return 1

def catch_regions(form: np.ndarray):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = cv2.dilate(form, kernel, iterations=1)
    q_w = 850
    q_h = 200
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
    # Recortando as regiões e guardando em uma lista
    regions = []
    regions.append(img[q1_i:q1_i+q_h, q_w:])
    regions.append(img[q2_i:q2_i+q_h, q_w:])
    regions.append(img[q3_i:q3_i+q_h, q_w:])
    regions.append(img[q4_i:q4_i+q_h, q_w:])
    regions.append(img[q5_i:q5_i+q_h, q_w:])
    regions.append(img[q6_i:q6_i+q_h, q_w:])

    regions.append(img[q7_i:q7_i+q_h, 0:1600])
    regions.append(img[q8_i:q8_i+300])
    regions.append(img[rating_i:rating_i+q_h, 700:])

    return regions

# Faz analise do formulário do tipo 0.
def analyze_form_0(imgs: list):
    results = []
    for img in imgs[:5]:
        results.append(analyze_question_type_0(img))


def analyze_question_type_0(img: np.ndarray) -> int:
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

    if exc_black > 2000:
        return 0
    elif good_black > 1300:
        return 1
    elif fair_black > 1000:
        return 2
    elif poor_black > 1000:
        return 3
    else:
        print("Não reconhecido")
        plt.imshow(img, cmap='gray')
        plt.show()
        return -1

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

    catch_forms_data(forms)


if __name__ == '__main__':
    main()
