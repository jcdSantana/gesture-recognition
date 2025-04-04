from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
import pandas as pd
import joblib

# Load dataset
dataset_path = "../data/test_processed/train.csv"
df = pd.read_csv(dataset_path)

# Separate features and labels
X = df.iloc[:, 1:].values  
y = df.iloc[:, 0].values   

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Create MLP model
model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=500, random_state=1)

# Train model
model.fit(X_train, y_train)

# Evaluate model
pred = model.predict(X_test)
recall = recall_score(y_test, pred, average='macro')
precision = precision_score(y_test, pred, average='macro')

print(f"Model recall: {recall:.4f}")
print(f"Model precision: {precision:.4f}")

# Save trained model
model.fit(X, y)
model_path = "../models/hand_gesture_mlp.pkl"
joblib.dump(model, model_path)
print(f"Model saved at {model_path}!")
