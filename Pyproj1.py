import sklearn
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, r2_score
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:\\Users\\Admin\\Downloads\\insurance.csv")
print(df.head())
#-------------------------------------------------------
print("-"*30)
print(df.info()) #Gives information about the columns
print(df.describe()) # Describes the data. Mean, Median, Mode etc...
#-------------------------------------------------------
print("-"*30)
print(df.isnull().sum()) # Gives the number of Null or missing values
#-------------------------------------------------------

print("-"*30)
#Now let's check if the cost increases based on different factors.

x=df[["sex","charges"]].groupby(["sex"],as_index=False).mean().sort_values(by="charges",ascending=False)
print(x) # Gives avg. charges for Male and Female.
print("-"*30)

# Similarly we find for Smoker and Non-Smoker.
print(df[["smoker","charges"]].groupby(["smoker"],as_index=False).mean().sort_values(by="charges",ascending=False))
print("-"*30)

#And Region
print(df[["region","charges"]].groupby(["region"],as_index=False).mean().sort_values(by="charges",ascending=False))

#-------------------------------------------------------

# Plot to show number of smokers and non-smokers that are male and female.

sns.catplot(x="smoker", kind="count",hue = 'sex', palette="icefire_r", data=df)
plt.subplots_adjust(top=0.9)
plt.title("The number of smokers and non-smokers")
plt.show()
#-------------------------------------------------------

sns.catplot(y="charges",x="smoker",hue ='sex', palette="spring_r", data=df[(df.age<35)])
plt.subplots_adjust(top=0.9)
plt.title("The charges for smokers and non-smokers(Male and Female) age < 35")
plt.show()

#-------------------------------------------------------

sns.catplot(y="charges",x="smoker",hue ='sex', palette="rainbow", data=df[(df.age>=35)])
plt.subplots_adjust(top=0.9)
plt.title("The charges for smokers and non-smokers(Male and Female) age >= 35")
plt.show()

#-------------------------------------------------------

plt.figure(figsize=(12,5))
plt.title("Box plot for charges for Male and Female")
sns.boxplot(y="charges", x="sex", data = df , palette = 'twilight_r')
plt.show()

#-------------------------------------------------------
# Scatter plot to show relationship between bmi and charges.
print("-"*30)
plt.figure(figsize=(10,6))
ax = sns.scatterplot(x='bmi',y='charges',data=df,palette='magma',hue='smoker')
ax.set_title('Scatter plot of charges and bmi')

# Scatter plot with a regression line.
sns.lmplot(x="bmi", y="charges", hue="smoker", data=df, palette = 'magma', height = 8)
plt.show()

#-------------------------------------------------------
# Create Dummies for Categorical Variables

l = LabelEncoder()
# sex
l.fit(df.sex.drop_duplicates()) 
df.sex = l.transform(df.sex)

# smoker or not
l.fit(df.smoker.drop_duplicates()) 
df.smoker = l.transform(df.smoker)

# region
l.fit(df.region.drop_duplicates()) 
df.region = l.transform(df.region)
print(df.head()) # Check Dummies
print("-"*30)

print(df.corr()) # Find Correlation among variables.
print("-"*30)
#-------------------------------------------------------
# Multivariate Linear Regression with Region

X=df.drop(["charges"],axis=1).copy() # Define dependent and independent variable 
Y=df.charges
lr = LinearRegression()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0) # Split data into training and test set (80:20)
train=lr.fit(X_train,Y_train)
pred=lr.predict(X_test)

print("Prediction Score: ",r2_score(Y_test,pred)) # Accuracy
print("-"*30)
#-------------------------------------------------------
# Multivariate Linear Regression without Region since correlation is almost negligible.

X1=df.drop(["charges","region"],axis=1).copy() 
Y1=df.charges
lr = LinearRegression()
X1_train,X1_test,Y1_train,Y1_test=train_test_split(X1,Y1,test_size=0.2,random_state=0)
train=lr.fit(X1_train,Y1_train)
pred2=lr.predict(X1_test)

print("Prediction Score: ",r2_score(Y1_test,pred2)) # Accuracy
print("-"*30)
#-------------------------------------------------------
# RandomForest Regressor
rf = RandomForestRegressor()
X2_train,X2_test,Y2_train,Y2_test=train_test_split(X,Y,random_state=0)
rf.fit(X2_train,Y2_train)
tr_pred = rf.predict(X2_train)
te_pred = rf.predict(X2_test)

print("R2 score",r2_score(Y2_train,tr_pred)) # Accuracy on training set
print("R2 Score",r2_score(Y2_test,te_pred)) # Accuracy on test set
