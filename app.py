# -*- coding: utf-8 -*-
"""
@author: Vlad Lee
"""

import streamlit as st
import pandas as pd
import altair as alt
import snowflake.connector
import matplotlib.pyplot as plt
import numpy as np
import streamlit_option_menu
from streamlit_option_menu import option_menu

#Define connection parameters
connection_params = {
    "user": "VLAD7984",
    "password": "Mgv1224?",
    "account": "hia45134.east-us-2.azure",  # E.g., "abc123.snowflakecomputing.com"
    "warehouse": "COMPUTE_WH",
    "database": "AVIATION_BIRD_STRIKES",
    "schema": "PUBLIC",
    "role": "ACCOUNTADMIN"
}

conn = snowflake.connector.connect(**connection_params)
cursor = conn.cursor()    
############################################################################
#Query to get total metrics
query_total_metrics = """
SELECT
    COUNT(*) AS total_bird_strikes,
    SUM(COST_REPAIRS + COST_OTHER) AS total_cost,
    SUM(COST_REPAIRS_INFL_ADJ + COST_OTHER_INFL_ADJ) AS total_cost_infl_adj,
    SUM(TRY_CAST(NR_FATALITIES AS NUMERIC)) AS total_fatalities,
    SUM(CAST(NR_INJURIES AS NUMERIC)) AS total_injuries
FROM FAA_BIRD_STRIKES
"""
cursor.execute(query_total_metrics)

#Fetch the results
metrics_result = cursor.fetchone()
total_bird_strikes = metrics_result[0]
total_cost = metrics_result[1]
total_cost_infl_adj = metrics_result[2]
total_fatalities = metrics_result[3]
total_injuries = metrics_result[4]

#Query to get top 10 states
query_top_states = """
SELECT 
    STATE, COUNT(*) AS bird_strikes
FROM FAA_BIRD_STRIKES
GROUP BY STATE
ORDER BY bird_strikes DESC
LIMIT 10
"""

#Execute query
cursor.execute(query_top_states)
top_states = cursor.fetchall()

#Create top states df
states_df = pd.DataFrame(top_states, columns=["State", "Bird Strikes"])
#Replace na with unknown
states_df['State'].fillna('UNKNOWN', inplace=True)

#Query to get top 10 airports
query_top_airports = """
SELECT 
    AIRPORT, COUNT(*) AS bird_strikes
FROM FAA_BIRD_STRIKES
GROUP BY AIRPORT
ORDER BY bird_strikes DESC
LIMIT 10
"""
#Execute query
cursor.execute(query_top_airports)
top_airports = cursor.fetchall()

#Create top airports df
airports_df = pd.DataFrame(top_airports, columns=["Airport", "Bird Strikes"])

############################################################################
#Configure settings for streamlit app
st.set_page_config(
    page_title="Aviation Bird Strikes",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#Configure main sidebar
with st.sidebar:
  selected = option_menu(
    menu_title = "Menu",
    options = ["Home","Bird Strikes Over Time", "Seasonality","State Map", "example3", "About"],
    icons = ["house","bird","seasons", "map","map","chart-simple"],
    menu_icon = "cast",
    default_index = 0,
  )
  
###################################################################################################
#Configure Home page.
if selected == "Home":
    st.title("Welcome to the Home Page")
    st.write("The purpose of this app is to allow users to analyze and visualize aviation bird strike data from 1996 \
             to 2024 and offer valuable \
             insights into trends. Select an option from the sidebar to get started.")
    st.markdown("## Key Metrics (1996 - 2024):")         
    
    #Create 4 columns for displaying the metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(label="Total Bird Strikes", value=f"{total_bird_strikes:,}")
    
    with col2:
        st.metric(label="Total Cost", value=f"${total_cost:,.2f}")
        
    with col3:
        st.metric(label="Total Cost (Inflation Adjusted)", value=f"${total_cost_infl_adj:,.2f}")
    
    with col4:
        st.metric(label="Total Fatalities", value=total_fatalities)
    
    with col5:
        st.metric(label="Total Injuries", value=total_injuries)
    
    #Create two columns for the bar charts side by side
    chart_col1, chart_col2 = st.columns(2)    
    
    with chart_col1:
        #Plot Top 10 States bar chart
        st.subheader("Top 10 States with Bird Strikes")
        fig, ax = plt.subplots()
        ax.bar(states_df["State"], states_df["Bird Strikes"], color='skyblue')
        ax.set_xlabel("State")
        ax.set_ylabel("Number of Bird Strikes")
        ax.set_title("Top 10 States by Bird Strikes")
        st.pyplot(fig)

    with chart_col2:
        #Plot Top 10 Airports bar chart
        st.subheader("Top 10 Airports with Bird Strikes")
        fig, ax = plt.subplots()
        ax.barh(airports_df["Airport"], airports_df["Bird Strikes"], color='lightgreen')
        ax.set_xlabel("Airport")
        ax.set_ylabel("Number of Bird Strikes")
        ax.set_title("Top 10 Airports by Bird Strikes")
        ax.invert_yaxis()
        st.pyplot(fig)

###################################################################################################
#Configure About page.
elif selected == "About":
    st.title("About")
    st.markdown("""
        My name is Vlad Lee. I am an economic consultant and a fellow at NYC Data Science Academy. 
        Feel free to check out my profile pages and GitHub!

        <p><a href='https://www.linkedin.com/in/vlad-lee' target='_blank'>LinkedIn</a></p>
        <p><a href='https://www.nera.com/experts/l/vladislav-lee.html?lang=en' target='_blank'>NERA</a></p>
        <p><a href='https://github.com/bomimyhomie/NFL-Analysis' target='_blank'>GitHub</a></p>
        <p>For questions or feedback, contact the author at 
        <a href='mailto:Vlad7984@gmail.com'>vlad7984@gmail.com</a>.</p>
    """, unsafe_allow_html=True)
