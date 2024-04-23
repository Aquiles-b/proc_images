import sys
from glob import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


# Retorna o tipo do formulário.
def catch_form_type(form: np.ndarray) -> int:
    # (912, 17, 606, 156)
    img_guess_f = form[16:175, 900:1520]
    bp = np.sum(img_guess_f <= 127)

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
    q8_9_i = 2100
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

    regions.append(img[q7_i:q7_i+q_h, q_w:1500])
    regions.append(img[q8_9_i:q8_9_i+q_h, 0:750])
    regions.append(img[q8_9_i:q8_9_i+q_h, 1300:2050])
    regions.append(img[rating_i:rating_i+q_h, 900:2380])

    return regions

# Retorna a quantidade de pixels pretos de cada região da questão.
def count_question_black_pixels(img: np.ndarray) -> tuple[int, int, int, int]:
    offset = 445
    img_exc = img[:, 0:offset]
    img_good = img[:, offset:offset*2]
    img_fair = img[:, offset*2:offset*3]
    img_poor = img[:, offset*3:]

    # count the number of black pixels
    exc_black = int(np.sum(img_exc <= 127))
    good_black = int(np.sum(img_good <= 127))
    fair_black = int(np.sum(img_fair <= 127))
    poor_black = int(np.sum(img_poor <= 127))

    return exc_black, good_black, fair_black, poor_black

# Retorna o indice da escolha feita na questão passada do formulário tipo 1.
def analyze_question_type_10(img: np.ndarray) -> int:
    exc_black, good_black, fair_black, poor_black = count_question_black_pixels(img)

    if exc_black >= 3300:
        return 0
    elif good_black >= 2200:
        return 1
    elif fair_black >= 2230:
        return 2
    elif poor_black >= 1500:
        return 3
    else:
        dis = []
        dis.append(abs(3300 - exc_black))
        dis.append(abs(2200 - good_black))
        dis.append(abs(2230 - fair_black))
        dis.append(abs(1500 - poor_black))

        return dis.index(min(dis))

# Retorna o indice da escolha feita na questão passada do formulário tipo 0.
def analyze_question_type_00(img: np.ndarray) -> int:
    exc_black, good_black, fair_black, poor_black = count_question_black_pixels(img)

    if exc_black >= 2000:
        return 0
    elif good_black >= 1300:
        return 1
    elif fair_black >= 1000:
        return 2
    elif poor_black >= 1000:
        return 3
    else:
        dis = []
        dis.append(abs(2000 - exc_black))
        dis.append(abs(1300 - good_black))
        dis.append(abs(1000 - fair_black))
        dis.append(abs(1000 - poor_black))

        return dis.index(min(dis))

# Retorna o indice da escolha feita na questão binária passada do formulário tipo 0.
def analyze_question_type_01(img: np.ndarray) -> int:
    _, width = img.shape
    img_yes = img[:, :width//2]
    img_no = img[:, width//2:]

    yes_black = int(np.sum(img_yes <= 127))
    no_black = int(np.sum(img_no <= 127))
    
    if yes_black >= 1100:
        return 0
    elif no_black >= 1100:
        return 1
    else:
        dis = []
        dis.append(abs(1100 - yes_black))
        dis.append(abs(1100 - no_black))

        return dis.index(min(dis))

# Retorna o indice da escolha feita na questão binária passada do formulário tipo 1.
def analyze_question_type_11(img: np.ndarray) -> int:
    _, width = img.shape
    img_yes = img[:, :width//2]
    img_no = img[:, width//2:]

    yes_black = int(np.sum(img_yes <= 127))
    no_black = int(np.sum(img_no <= 127))

    if yes_black >= 1700:
        return 0
    elif no_black >= 1700:
        return 1
    else:
        dis = []
        dis.append(abs(1700 - yes_black))
        dis.append(abs(1700 - no_black))

        return dis.index(min(dis))

# Retorna a nota da questão de avaliação.
def analyze_rating(img: np.ndarray) -> int:
    _, width = img.shape
    offset = width // 10

    values = []

    for i in range(10):
        values.append(int(np.sum(img[:, offset*i:offset*(i+1)] <= 127)))

    return values.index(max(values))

# Faz analise do formulário do tipo 0.
def analyze_form_0(imgs: list) -> list:
    results = []
    for img in imgs[:6]:
        results.append(analyze_question_type_00(img))
    for img in imgs[6:9]:
        results.append(analyze_question_type_01(img))

    results.append(analyze_rating(imgs[9]))

    return results

# Faz analise do formulário do tipo 1.
def analyze_form_1(imgs: list) -> list:
    results = []
    for img in imgs[:6]:
        results.append(analyze_question_type_10(img))
    for img in imgs[6:9]:
        results.append(analyze_question_type_11(img))

    results.append(analyze_rating(imgs[9]))

    return results

def tag_answers_img(form_path: str, answers: list, form_type: int, out_dir: str) -> None:
    pass

# Retorna uma lista com os dados dos formulários.
def catch_forms_data(forms: list, tag_out_dir: str) -> list:
    results = [[0]*4 for _ in range(6)]
    results.append([0]*2)
    results.append([0]*2)
    results.append([0]*2)
    results.append([0]*10)

    for form in forms:
        img = cv2.imread(form, cv2.IMREAD_GRAYSCALE)
        form_type = catch_form_type(img)

        regions = catch_regions(img)

        if form_type == 1:
            r = analyze_form_1(regions)
        else:
            r = analyze_form_0(regions)

        for idx, i in enumerate(r):
            results[idx][i] += 1

        if tag_out_dir != "":
            tag_answers_img(form, r, form_type,tag_out_dir)

    return results

def save_results(results: list, n: int) -> None:
    r0 = np.array(results[:6])
    r1 = np.array(results[6:9])
    r2 = np.array(results[9])

    r0 = ((r0 / n)*100).astype(int)
    r1 = ((r1 / n)*100).astype(int)
    media = (np.sum(r2 * np.arange(10)) / n) * 10
    media = int(media)

    linhas = ["".join([f"{i} " for i in l]) + "\n" for l in r0]
    linhas += ["".join([f"{i} " for i in l]) + "\n" for l in r1]
    tw = "".join(linhas)
    tw += f"{media}"

    with open("results.txt", "w") as f:
        f.write(tw)

def main() -> None:
    argc = len(sys.argv)
    if argc < 2 or argc > 3:
        print("Usage: python3 forms.py <forms-dir> <forms-out-dir>")
        sys.exit(1)

    forms_dir = sys.argv[1]
    if not os.path.exists(forms_dir):
        print(f"Error: {forms_dir} does not exist.")
        sys.exit(1)

    forms = sorted(glob(f"{forms_dir}/*.png"))
    if len(forms) == 0:
        print(f"Error: {forms_dir} does not contain any forms.")
        sys.exit(1)

    tag_out_dir = ""
    if argc == 3:
        tag_out_dir = sys.argv[2]
        os.makedirs(tag_out_dir, exist_ok=True)

    results = catch_forms_data(forms, tag_out_dir)
    save_results(results, len(forms))


if __name__ == '__main__':
    main()
