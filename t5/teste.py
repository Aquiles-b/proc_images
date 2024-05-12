from make_all_tests import train_LBP_classifier
from generate_data import create_LBP_csv_data
from eval import evaluate_clf_LBP
import numpy as np

create_LBP_csv_data((816, 612))
model = train_LBP_classifier(100, lr=0.001)
cm = evaluate_clf_LBP(model, 'val', 9)
acc = np.trace(cm) / np.sum(cm)
print(acc)

