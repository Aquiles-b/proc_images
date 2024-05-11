from core import read_hists, KNN_decision
import sys


if sys.argv[1] == "lbp":
    train_hists = read_hists("data/train_LBP.csv")
    val_hists = read_hists("data/val_LBP.csv")
else:
    train_hists = read_hists("data/train_CNN.csv")
    val_hists = read_hists("data/val_CNN.csv")

confusion_matrix = [[0] * 9 for _ in range(9)]

correct, miss = 0, 0
for hist in val_hists:
    decision = KNN_decision(hist, train_hists, knn=1, num_classes=9)
    lbl = int(hist[-1])
    confusion_matrix[lbl][decision] += 1
    if lbl == decision:
        correct += 1
    else:
        miss += 1

print(f'{(correct / (correct + miss)):.2f}')
for i, line in enumerate(confusion_matrix):
    total = sum(line)
    print(line, f"{line[i]/total:.2f}")


