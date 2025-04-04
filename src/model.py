from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
import pandas as pd
import numpy as np
import joblib

# Load dataset
dataset_path = "../data/test_processed/test.csv"
df = pd.read_csv(dataset_path)

# Separate features and labels
X = df.iloc[:, 1:].values  
y = df.iloc[:, 0].values   

# Normalize data using Min-Max Scaling
X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create KNN model
model = KNeighborsClassifier(n_neighbors=5)  
# Train model
model.fit(X_train, y_train)

# Evaluate model
pred = model.predict(X_test)
recall = recall_score(y_test, pred, average='micro')
precission = precision_score(y_test, pred, average='micro')

print(f"Model recall: {recall:.4f}")
print(f"Model precission: {precission:.4f}")


# Save trained model
model_path = "../models/hand_gesture_knn.pkl"
joblib.dump(model, model_path)
print(f"Model saved at {model_path}!")