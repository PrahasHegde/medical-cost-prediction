#USING RANDOM FOREST REGRESSOR

#imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error



df = pd.read_csv("insurance.csv")

#more info about dataset
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())

#separating categorical and numerical cols
cat_cols = [col for col in df.columns if df[col].dtype == 'object']
num_cols = [col for col in df.columns if df[col].dtype in ['int', 'float']]

print(df[cat_cols].nunique())

#replaced all the categorical values with numbers.
df.replace({'sex':{'male':1, 'female':0}}, inplace=True)
df.replace({'smoker':{'yes':1, 'no':0}}, inplace=True)
df.replace({'region':{'southeast':1, 'southwest':2, 'northwest':3, 'northeast':4}}, inplace=True)

#correlation Heatmap
sns.heatmap(df.corr())
plt.show()

#check the dataset after operation
# print(df.head().info())

#Train-Test Split
X = df.drop(columns=['charges'])
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=897)

#Models Building and Evaluation

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_prediction = rf.predict(X_test)  
print(f"MAE: {mean_absolute_error(y_test, rf_prediction)}, \nR2-Score: {r2_score(y_test, rf_prediction)}")


#plot actual vs predicted
actual = y_test
predicted = rf_prediction


plt.figure(figsize=(15, 10))

# Plot the actual values as a scatter plot
plt.scatter(range(len(actual)), actual, color='blue', label='Actual')

# Plot the predicted values as a line
plt.scatter(range(len(actual)), predicted, color='red', label='Predicted')

# A line between the actual point and predicted point
for i in range(len(actual)):
    plt.plot([i, i], [actual.iloc[i], predicted[i]], color='green', linestyle='--')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values (medical cost prediction)')
plt.legend()
plt.show()