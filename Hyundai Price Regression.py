#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import chi2


# In[2]:


df=pd.read_csv(r'C:\Users\KrishnaBalram\Desktop\masai\ML\datasets\hyundi.csv')
df.head()


# ### size

# In[3]:


df.shape


# ### treat missing values

# In[4]:


df.isna().sum()


# ### Check column types and describe which columns are numerical or categorical

# In[5]:


df.dtypes


# In[6]:


df.price.describe()


# In[7]:


categorical=df.describe(include='object').columns
categorical


# In[8]:


numerical=df.describe(include=[np.number]).columns
numerical


# In[9]:


ls= df.columns

for i in ls:
    print('-----------------')
    print(i)
    print(df[i].value_counts())


# In[10]:


#MOdel Count

plt.figure(figsize=(12,10))
sns.countplot(y='model', data=df)
plt.title('Model types')
plt.show()


# In[11]:


model_price = df.groupby('model')['price'].mean().sort_values()

plt.figure(figsize=(14, 8))
plt.title("Hyundai Average Price for each Model")
pal = sns.color_palette("Greens_d", len(model_price))

sns.barplot(x=model_price.index, y=model_price.values, palette=pal)

plt.xlabel("Model")
plt.ylabel("Price (Euros)")
plt.tight_layout()
plt.savefig("output.png")


# In[12]:


year_price = df.groupby('year')['price'].mean().sort_values()

plt.figure(figsize=(14, 8))
plt.title("Hyundai Average Price by Year")
pal = sns.color_palette("Greens_d", len(year_price))

sns.barplot(x=year_price.index, y=year_price.values, palette=pal)

plt.xlabel("Year")
plt.ylabel("Price (Euros)")
plt.tight_layout()
plt.savefig("output.png")


# In[ ]:





# In[ ]:





# In[13]:


# Higher Transmission Preference 
sns.countplot(x='transmission', data=df)
plt.title('Transmission Types')
plt.show()


# In[14]:


# Most Fuel type preference 

sns.countplot(x='fuelType', data=df)
plt.title('Fuel Types')
plt.show()


# In[15]:


#Visualizing categorical data columns
fuelType = df['fuelType']
transmission = df['transmission']
price = df['price']
fig, axes = plt.subplots(1,2, figsize=(15,5), sharey=True)
fig.suptitle('Visualizing categorical data columns')
sns.barplot(x=fuelType, y=price, ax=axes[0])
sns.barplot(x=transmission, y=price, ax = axes[1])


# ### Univariate analysis

# ### 1.  Calculate mean, median, std dev, and quartiles

# In[16]:


df.describe(include=[np.number])


# ### Plot histogram for a few categorical variables

# In[17]:


for i in categorical:
    plt.figure(figsize=(12, 6)) 
    sns.histplot(df[i]) 
    plt.show()


# ### Check the distribution of numerical variables and comment on it

# In[18]:


for i in numerical:
    print(i)       
    sns.kdeplot(df[i])
    plt.show()
    
    print('----------------------------------------------------') 


# In[19]:


plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
sns.scatterplot(x=df['tax(£)'],y=df.price)

plt.subplot(1,3,2)
sns.scatterplot(x=df.mpg,y=df.price)

plt.subplot(1,3,3)
sns.scatterplot(x=df.mileage,y=df.price)


# Mileage and mpg seem to have a negative correlation with price.
# Tax doesn't show any trend.

# ### Bivariate analysis

# ### Plot pair plots

# In[20]:


sns.pairplot(df, hue ='transmission')
plt.show()


# ### Chi-square analysis

# ### transmission and fuelType

# In[21]:


from scipy.stats import chi2_contingency
  
# defining the table
data = pd.crosstab(df['transmission'],df['fuelType'])
stat, p, dof, expected = chi2_contingency(data)
  
# interpret p-value
alpha = 0.05
print("p value is " + str(p))
if p <= alpha:
    print('cols r Dependent (reject H0)')
else:
    print('cols r Independent (H0 holds true)')


# ### fuelType and model

# In[22]:


from scipy.stats import chi2_contingency
  
# defining the table
data = pd.crosstab(df['fuelType'],df['model'],)
stat, p, dof, expected = chi2_contingency(data)
  
# interpret p-value
alpha = 0.05
print("p value is " + str(p))
if p <= alpha:
    print('cols r Dependent (reject H0)')
else:
    print('cols r Independent (H0 holds true)')


# ### Pearson correlation, and plot their heatmap

# In[23]:


plt.figure(figsize=(16, 6))
sns.heatmap(df.corr(), cmap="YlGnBu", annot=True,vmin=-1, vmax=1)
plt.show()


# In[24]:


#Correlation between year and price

fig = plt.figure(figsize=(7,5))
plt.title('Correlation between year and price')
sns.regplot(x='price', y='year', data=df)


# ### Drop any unnecessary columns

# In[25]:


df.drop(['model'], axis=1,inplace=True)


# ### One hot encode categorical variables 

# In[26]:


import category_encoders as ce
enc = ce.OneHotEncoder(cols=['transmission','fuelType'], return_df=True)
df = enc.fit_transform(df)


# In[27]:


df = df.reset_index(drop=True)


# In[28]:


df.head()


# In[29]:


df.shape


# ### Split into train and test set

# In[30]:


x = df.drop('price',axis=1)
y = df['price']


# In[31]:


print(x.shape)
print(y.shape)


# In[32]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=51)


# #### Scale the variables`

# In[33]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
inputs = ['year','price','mileage','tax(£)','mpg','engineSize']
df[inputs] = mms.fit_transform(df[inputs])
df.head()


# ## Linear regression, Decision Tree, Random Forest, SVR

# In[34]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


# ### TRAING DATA

# In[36]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
def model_train(x_train,y_train):
  all_models = [LinearRegression(),RandomForestRegressor(),DecisionTreeRegressor(),SVR()]
  scores = []
  for i in all_models:
    model = i
    model.fit(x_train,y_train)
    y_predicted = model.predict(x_train)
    mse = mean_squared_error(y_train,y_predicted)
    mae = mean_absolute_error(y_train,y_predicted)
    
    scores.append({
        'model': i,
        'best_score':model.score(x_train,y_train),
        'mean_squared_error':mse,
        'mean_absolute_error':mae
    })
  return pd.DataFrame(scores,columns=['model','best_score','mean_squared_error','mean_absolute_error'])

model_train(x_train,y_train)


# ### TESTING DATA

# In[37]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
def model_test(x_test,y_test):
  all_models = [LinearRegression(),RandomForestRegressor(),DecisionTreeRegressor(),SVR()]
  scores = []
  for i in all_models:
    model = i
    model.fit(x_test,y_test)
    y_predicted = model.predict(x_test)
    mse = mean_squared_error(y_test,y_predicted)
    mae = mean_absolute_error(y_test,y_predicted)
    scores.append({
        'model': i,
        'best_score':model.score(x_test,y_test),
        'mean_squared_error':mse,
        'mean_absolute_error':mae
        })
  return pd.DataFrame(scores,columns=['model','best_score','mean_squared_error','mean_absolute_error'])


model_test(x_test,y_test)


# ### Based on Above Observation we can say Decision Tree model is the best model for our data

# ### Visualization Of Predicted Model

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(x_test,y_test)
y_predicted = model.predict(x_test)

plt.figure(figsize=(10,8))
plt.title('Truth vs Predicted',fontsize=25)
sns.scatterplot(x = y_test,y = y_predicted)      
plt.xlabel('Truth', fontsize=18)                          
plt.ylabel('Predicted', fontsize=16)   


# ## Hypertuning Grid search CV

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = {  'bootstrap': [True], 'max_depth': [5,10, None], 'max_features': ['auto', 'log2'],
              'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]}


# In[ ]:


rfr = RandomForestRegressor(random_state = 1)

g_search = GridSearchCV(estimator = rfr, param_grid = param_grid, 

                          cv = 3, n_jobs = 1, verbose = 0, return_train_score=True)


# In[ ]:


g_search.fit(x_train, y_train);

print(g_search.best_params_)


# In[ ]:


print(g_search.score(x_test, y_test))


# # Train a polynomial regression model with degrees 2, and 3 and compare its performance with other models 
# 

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=True)
x_train_trans = poly.fit_transform(x_train)
x_test_trans = poly.transform(x_test)
lr = LinearRegression()
lr.fit(x_train_trans, y_train)
y_pred = lr.predict(x_test_trans)
print(r2_score(y_test, y_pred))


# In[ ]:


poly = PolynomialFeatures(degree=3, include_bias=True)
x_train_trans = poly.fit_transform(x_train)
x_test_trans = poly.transform(x_test)
lr = LinearRegression()
lr.fit(x_train_trans, y_train)
y_pred = lr.predict(x_test_trans)
print(r2_score(y_test, y_pred))


# In[ ]:




