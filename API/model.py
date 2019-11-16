# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

data=pd.read_csv('/Users/apple/Documents/Deployment-flask/new_data_actual.csv')

data_points = pd.DataFrame()#dataframe

data_points["Depth"] = data["Depth"]
data_points["WS"] = data["WS"]
data_points["RH (%)"] = data["RH (%)"]
data_points["Temp (Deg)"] = data["Temp (Deg)"]

data_points = np.array(data_points)
#1st depedent variable

data_target1 = pd.DataFrame(data["PM0.23_0.30"])
data_target1 = np.array(data_target1)
data_target1 = data_target1.reshape(1,-1)[0]
#imports random forest


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_points, data_target1, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestRegressor   
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(data_points, data_target1)

# Saving model to disk
pickle.dump(base_model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6,4]]))
