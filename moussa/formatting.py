import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned and formatted dataset (from Step 1)
df = pd.read_csv("../moussa/dataset_ready_for_ml.csv")  # Adjust path if needed

# 1. Separate features (X) and target (y)
# Replace 'price' by the actual target column in your dataset
X = df.drop(columns=["price"])
y = df["price"]

# 2. Split into training and testing sets
# We'll use 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Confirm split shape
print("✅ Data successfully split for ML:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
