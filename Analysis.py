# -*- coding: utf-8 -*-
import pandas as pd
import snowflake.connector
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Define connection parameters
connection_params = {
    "user": "",
    "password": "",
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
#Simple linear regression y = cost
y_simple = bird_strikes_df['TOTAL_COST']
x_simple = bird_strikes_df['HEIGHT']
x_simple = x_simple.dropna()

#Align indices and drop na values
y_simple = y_simple.loc[x_simple.index].dropna()
#Add constant
x_simple = sm.add_constant(x_simple)

#Run OLS regression
model_simple = sm.OLS(y_simple, x_simple)
results_simple = model_simple.fit()
reg_summary_simple = results_simple.summary()

print(results_simple.summary())

#Plot residuals
plot_residual_diagnostics(results_simple)

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

print(results2.summary())

#Plot residuals
fitted_values2 = results2.fittedvalues
residuals2 = results2.resid
plt.figure(figsize=(8,6))
sns.scatterplot(x=fitted_values2, y=residuals2, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')  # Horizontal line at 0
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted")
plt.show()


