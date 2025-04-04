import joblib
from sklearn.metrics import recall_score, precision_score
import pandas as pd
model_path = "../models/hand_gesture_knn.pkl"
model = joblib.load(model_path)

dataset_path = "../data/test_processed/test.csv"
df = pd.read_csv(dataset_path)

X = df.iloc[:, 1:].values  
y = df.iloc[:, 0].values   

pred = model.predict(X)
recall_macro = recall_score(y, pred, average='macro')
precission_macro = precision_score(y, pred, average='macro')

recall_micro = recall_score(y, pred, average='micro')
precission_micro = precision_score(y, pred, average='micro')

print(f"Model recall macro: {recall_macro:.4f}")
print(f"Model precission macro: {precission_macro:.4f}")
print(f"Model recall micro: {recall_micro:.4f}")
print(f"Model precission micro: {precission_micro:.4f}")
