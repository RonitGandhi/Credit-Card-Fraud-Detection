# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Loading the dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('creditcard.csv')

# Extracting features (X) and target variable (y)
X = data.drop(['Amount', 'Class'], axis=1)  # Assuming 'Amount' is the target variable
y = data['Amount']

data.dropna(inplace=True)
X_cleaned = X.dropna()
y_cleaned = y[X.index.isin(X_cleaned.index)]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# Initializing the linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

import matplotlib.pyplot as plt
import seaborn as sns

# 目标变量 'Amount' 的分布
plt.figure(figsize=(10, 6))
sns.histplot(data['Amount'], bins=30, kde=True, color='blue')
plt.title('Distribution of Credit Limits')
plt.xlabel('Credit Limit')
plt.ylabel('Frequency')
plt.show()

# 不同特征与目标变量 'Amount' 的关系
# 以 'V1' 为例
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Class'], y=data['Amount'], color='green')
plt.title('Scatter Plot of Class vs. Credit Limit')
plt.xlabel('Class')
plt.ylabel('Credit Limit')
plt.show()

# 特征相关性热力图
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Features')
plt.show()

# 模型预测值与实际值的对比
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='purple')
plt.title('Actual vs. Predicted Credit Limits')
plt.xlabel('Actual Credit Limit')
plt.ylabel('Predicted Credit Limit')
plt.show()