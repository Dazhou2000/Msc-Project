import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# open feature dataset and label
data = pd.read_csv('features and seizure labels.csv')
importance = pd.read_csv('P1_perm_importance with window_576.csv')
top_10_features = importance.sort_values(by='Importance', ascending=False)
feature_names=top_10_features['Unnamed: 0'].tolist()

# length =len(data)
# train_length = int(length * 0.8)
# test_length = int(length * 0.1)
# gap_length = int(length * 0.1)
# if train_length + test_length + gap_length == length:
#     print('successful')
# else:
#     print('wrong segment')
X = data[feature_names]
print(X)
y = data['label']

# Split the data into training and testing sets
# X_train = X.iloc[:train_length]
# X_test = X.iloc[train_length + gap_length:]
# y_train = y.iloc[:train_length]
# y_test = y.iloc[train_length + gap_length:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# Train the SVM classifier
svm_model = LinearSVC( random_state=1,max_iter=10000)
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)
result = pd.DataFrame({'test':y_test , 'predict': y_pred})
result.to_csv('result.csv')

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate sensitivity (recall for class 1)
sensitivity = recall_score(y_test, y_pred, pos_label=1)

# Calculate specificity (recall for class 0)
specificity = recall_score(y_test, y_pred, pos_label=0)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Print the results
print(f'Confusion Matrix:\n{cm}')
print(f'Accuracy: {accuracy:.2f}')
print(f'Sensitivity: {sensitivity:.2f}')
print(f'Specificity: {specificity:.2f}')
