import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# import
df = pd.read_csv("case-studies\loan-approval\data\loan_approval_data.csv")

# split by features
x = df[['Age', 'Income', 'LoanAmount', 'CreditScore']]
y = df['Approved']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize XGBoost
model = XGBClassifier(
    learning_rate=0.1, 
    max_depth=3, 
    n_estimators=50,
    eval_metric='logloss'
    )

model.fit(X_train, Y_train)

# prediction
y_pred = model.predict(X_test)

# evaluation
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion matrix
cm = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix (raw numbers):")
cm_percent = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]

fig, ax = plt.subplots(figsize=(5,4))
cax = ax.matshow(cm_percent, cmap=plt.cm.Blues)
for i in range(cm_percent.shape[0]):
    for j in range(cm_percent.shape[1]):
        ax.text(j, i, f"{cm_percent[i,j]:.2f}", ha='center', va='center', color='black')
    
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(['Rejected', 'Approved'])
ax.set_yticklabels(['Rejected', 'Approved'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Percentage)')
plt.colorbar(cax)
plt.show()