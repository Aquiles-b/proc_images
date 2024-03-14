import sys
import cv2
import json
import os
from glob import glob


def calc_histogram(images: list) -> tuple[list, list]:
    hists = []
    classes = []
    num_classes = 5
    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        # hist = hist.flatten().tolist()
        classes.append(idx // num_classes)
        hists.append(hist)

    return hists, classes

def load_hists_json(json_name: str) -> tuple[list, list]:
    with open(json_name, 'r') as f:
        hists_json = json.load(f)

    classes = hists_json[0]
    hists = hists_json[1:]

    return hists, classes

def save_hists_json(json_name: str, hists: list, classes: list) -> None:
    with open(json_name, 'w') as f:
        json.dump([classes] + hists, f)

def classify_by_knn(method, data: list, base: list[list]) -> None:
    pass

if __name__ == "__main__":
    method = sys.argv[1]
    images_path = sys.argv[2]
    hist_json_path = f"{images_path}.json"
    images = sorted(glob(f"{images_path}/*"))

    if os.path.exists(hist_json_path):
        hists = load_hists_json(hist_json_path)
    else:
        hists, classes = calc_histogram(images)

    c = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL)
    print(c)
