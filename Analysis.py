# -*- coding: utf-8 -*-
import pandas as pd
import snowflake.connector
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

#Define connection parameters
connection_params = {
    "user": "",
    "password": "?",
    "account": "",  # E.g., "abc123.snowflakecomputing.com"
    "warehouse": "",
    "database": "",
    "schema": "",
    "role": ""
}

conn = snowflake.connector.connect(**connection_params)
cursor = conn.cursor()   

#Query to pull data from Snowflake
query = """
SELECT *
FROM FAA_BIRD_STRIKES
"""
cursor.execute(query)
bird_strikes = cursor.fetchall()
col_names = [desc[0] for desc in cursor.description]

#Create df
bird_strikes_df = pd.DataFrame(bird_strikes, columns=col_names)

#Convert to numeric
bird_strikes_df['HEIGHT'] = pd.to_numeric(bird_strikes_df['HEIGHT'], errors="coerce")
bird_strikes_df['SPEED'] = pd.to_numeric(bird_strikes_df['SPEED'], errors="coerce")
bird_strikes_df['DISTANCE'] = pd.to_numeric(bird_strikes_df['DISTANCE'], errors="coerce")
bird_strikes_df['TOTAL_COST'] = pd.to_numeric(bird_strikes_df['TOTAL_COST'], errors="coerce")

#################################################################################################
#Function to create residual diagnostic plots
def plot_residual_diagnostics(model):
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    #Residuals vs Fitted
    sns.scatterplot(x=fitted_values, y=residuals, alpha=0.5, ax=axes[0, 0])
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Fitted")

    #Q-Q Plot
    sm.qqplot(residuals, line='45', fit=True, ax=axes[0, 1])
    axes[0, 1].set_title("Q-Q Plot")

    #Scale-Location Plot
    sns.scatterplot(x=fitted_values, y=np.sqrt(abs(residuals)), alpha=0.5, ax=axes[1, 0])
    axes[1, 0].axhline(y=0, color='red', linestyle='--')
    axes[1, 0].set_xlabel("Fitted Values")
    axes[1, 0].set_ylabel("√|Residuals|")
    axes[1, 0].set_title("Scale-Location Plot")

    #Residuals vs Leverage
    sm.graphics.influence_plot(model, criterion="cooks", ax=axes[1, 1])
    axes[1, 1].set_title("Residuals vs Leverage")

    plt.tight_layout()
    plt.show()

#################################################################################################
#Multiple linear regression y = cost, x = height, speed, distance
#Set x and y variables
x = bird_strikes_df[['HEIGHT','SPEED', 'DISTANCE']]
#Drop na values
x = x.dropna(subset=['HEIGHT','SPEED','DISTANCE'])
y = bird_strikes_df['TOTAL_COST']

#Align indices and drop na values
y = y.loc[x.index].dropna()
#Add constant
x = sm.add_constant(x)

#Run OLS regression
model = sm.OLS(y, x)
results = model.fit()
reg_summary = results.summary()

print(results.summary())

#Create correlation matrix
correlation_matrix = x.corr()
print(correlation_matrix)

#Calculate variance inflation factor
vif_data = pd.DataFrame()
vif_data["Variable"] = x.columns
vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]

print("VIF Results:")
print(vif_data)

#Plot residuals
plot_residual_diagnostics(results)

#################################################################################################
#Drop distance and run regression
x2 = x[['HEIGHT', 'SPEED']]
y2 = y.loc[x.index].dropna()
x2 = sm.add_constant(x2)

#Run OLS regression
model2 = sm.OLS(y2, x2)
results2 = model2.fit()
reg_summary2 = results2.summary()

#Calculate variance inflation factor
vif_data2 = pd.DataFrame()
vif_data2["Variable"] = x2.columns
vif_data2["VIF"] = [variance_inflation_factor(x2.values, i) for i in range(x2.shape[1])]

print(results2.summary())

#Plot residuals
plot_residual_diagnostics(results2)

#################################################################################################
#Create df for random forest model
#Drop unnecessary columns
tree_df = bird_strikes_df.drop(['INDEX_NR', 'INCIDENT_DATE', 'INCIDENT_MONTH', 'INCIDENT_YEAR', 'TIME',
              'AIRPORT_LATITUDE', 'AIRPORT_LONGITUDE', 'STATE', 'FAAREGION', 'LOCATION', 
              'OPERATOR', 'FLT', 'AIRCRAFT', 'ENG_1_POS', 'ENG_2_POS', 'ENG_3_POS', 'ENG_4_POS', 'AOS',
              'COST_REPAIRS', 'COST_OTHER', 'COST_REPAIRS_INFL_ADJ', 'COST_OTHER_INFL_ADJ',
              'INDICATED_DAMAGE', 'OTHER_SPECIFY', 'SPECIES', 'OUT_OF_RANGE_SPECIES', 
              'NUM_SEEN', 'NUM_STRUCK', 'ENROUTE_STATE', 'NR_INJURIES', 'NR_FATALITIES', 'EFFECT', 'EFFECT_OTHER'
              ], axis=1)

#Remove special characters
tree_df["DAMAGE_LEVEL"] = tree_df["DAMAGE_LEVEL"].str.replace(r"[^A-Za-z0-9\s]", "", regex=True)
#Drop NAs
tree_df_clean = tree_df.dropna()
#Drop Distance
tree_df_clean = tree_df_clean.drop(['DISTANCE'], axis=1)

#Separate out continuous variables
continuous_var = tree_df_clean[['HEIGHT', 'SPEED', 'TOTAL_COST']]
categorical_var = tree_df_clean.drop(['HEIGHT', 'SPEED', 'TOTAL_COST'], axis=1)

#Label encoding for categorical variables
for col in categorical_var.columns:
    le = LabelEncoder()
    categorical_var.loc[:, col] = le.fit_transform(categorical_var[col])

#Log transformation of continuous variables
continuous_var.loc[:, 'HEIGHT'] = np.log(continuous_var['HEIGHT'] + 1)
continuous_var.loc[:, 'SPEED'] = np.log(continuous_var['SPEED'] + 1)
continuous_var.loc[:, 'TOTAL_COST'] = np.log(continuous_var['TOTAL_COST'] + 1)

#Separate target variable (Total Cost) and features
y_tree = continuous_var['TOTAL_COST']
x_tree = pd.concat([continuous_var.drop(['TOTAL_COST'], axis=1), categorical_var], axis=1) 
    
#Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_tree, y_tree, test_size=0.2, random_state=1)

#Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=500, min_samples_split=10, random_state=1)

#Fit the model to the training data
rf_model.fit(x_train, y_train)

#Make predictions on the test data
y_pred = rf_model.predict(x_test)

#Calculate R2, MAE, MSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#Print R2, MAE, MSE
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

#Test importance of features
feature_importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x_train.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

#Min and max of log total cost
print(continuous_var['TOTAL_COST'].min())
print(continuous_var['TOTAL_COST'].max())

#################################################################################################
#Drop variables with importance below 0.01 and rerun model
important_features = feature_importance_df[feature_importance_df['Importance'] >= 0.01]
#Keep only the important features as x variables
x_tree_filtered = x_tree[important_features['Feature']]

#Split data into train and test sets
x_train2, x_test2, y_train2, y_test2 = train_test_split(x_tree_filtered, y_tree, test_size=0.2, random_state=1)

#Initialize the Random Forest Regressor model
rf_model2 = RandomForestRegressor(n_estimators=500, min_samples_split=10, random_state=1)

#Fit the model to the training data
rf_model2.fit(x_train2, y_train2)

#Make predictions on the test data
y_pred2 = rf_model2.predict(x_test2)

#Calculate R2, MAE, MSE
mae2 = mean_absolute_error(y_test2, y_pred2)
mse2 = mean_squared_error(y_test2, y_pred2)
r2_new = r2_score(y_test2, y_pred2)

#Print R2, MAE, MSE
print(f"Mean Absolute Error (MAE): {mae2}")
print(f"Mean Squared Error (MSE): {mse2}")
print(f"R-squared (R2): {r2_new}")

#Test importance of features
feature_importances2 = rf_model2.feature_importances_
feature_importance_df2 = pd.DataFrame({'Feature': x_train2.columns, 'Importance': feature_importances2})
feature_importance_df2 = feature_importance_df2.sort_values(by='Importance', ascending=False)
print(feature_importance_df2)





