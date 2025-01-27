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
import folium
from folium.plugins import HeatMap
from io import BytesIO
from streamlit_folium import folium_static

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
###################################################################################################
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
    options = ["Home","Heatmap", "Yearly Strikes","Time of Day", "Phase of Flight","Seasonality", "About"],
    icons = ["house","map","bar_chart","clock","airplane_departure","map","chart-simple"],
    menu_icon = "cast",
    default_index = 0,
  )


###################################################################################################
#Query to get bird strikes grouped by time of day
query_ToD = """
SELECT 
    TIME_OF_DAY, COUNT(*) AS bird_strikes
FROM FAA_BIRD_STRIKES
GROUP BY TIME_OF_DAY
ORDER BY bird_strikes ASC
"""

#Execute query
cursor.execute(query_ToD)
time_of_day = cursor.fetchall()

#Create time of day df, convert strikes to numeric
time_of_day_df = pd.DataFrame(time_of_day, columns=["Time of Day", "Bird Strikes"])
time_of_day_df["Bird Strikes"] = pd.to_numeric(time_of_day_df["Bird Strikes"], errors="coerce")

#Convert na to unknown
time_of_day_df['Time of Day'].fillna('Unknown', inplace=True)

#Query to get bird_strikes by time
query_time = """
SELECT 
    HOUR(TIME), COUNT(*) AS bird_strikes
FROM FAA_BIRD_STRIKES
GROUP BY HOUR(TIME)
ORDER BY bird_strikes ASC
"""

#Execute query
cursor.execute(query_time)
time_hour = cursor.fetchall()

#Create time  df, convert strikes to numeric
time_df = pd.DataFrame(time_hour, columns=["Hour", "Bird Strikes"])
time_df["Bird Strikes"] = pd.to_numeric(time_df["Bird Strikes"], errors="coerce")
time_df = time_df.dropna()
  
###################################################################################################
#Configure Home page.
if selected == "Home":
    st.title("Bird Strikes in Aviation Dashboard")
    st.write("The purpose of this app is to allow users to analyze and visualize aviation bird strike data from 1996 \
             to present and offer valuable \
             insights into trends. Select an option from the sidebar to get started.")
    st.markdown("## Key Metrics (1996 - 2025):")         
    
    
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
    
###################################################################################################
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
    states_df['State'].fillna('Unknown', inplace=True)

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

    #Query to get top 10 carriers
    query_top_carriers = """
    SELECT 
        OPERATOR, COUNT(*) AS carriers
    FROM FAA_BIRD_STRIKES
    GROUP BY OPERATOR
    ORDER BY carriers DESC
    LIMIT 10
    """
    #Execute query
    cursor.execute(query_top_carriers)
    top_carriers = cursor.fetchall()

    #Create top airports df
    carriers_df = pd.DataFrame(top_carriers, columns=["Carrier", "Bird Strikes"])
    
    #Create two columns for the bar charts side by side
    chart_col1, chart_col2, chart_col3 = st.columns(3)    
    
    with chart_col1:
        #Plot Top 10 States bar chart
        st.subheader("Top 10 States with Bird Strikes")
        fig, ax = plt.subplots()
        ax.bar(states_df["State"], states_df["Bird Strikes"], color='skyblue')
        ax.set_xlabel("State")
        ax.set_ylabel("Number of Bird Strikes")
        st.pyplot(fig)

    with chart_col2:
        #Plot Top 10 Airports bar chart
        st.subheader("Top 10 Airports with Bird Strikes")
        fig, ax = plt.subplots()
        ax.barh(airports_df["Airport"], airports_df["Bird Strikes"], color='lightgreen')
        ax.set_xlabel("Airport")
        ax.set_ylabel("Number of Bird Strikes")
        ax.invert_yaxis()
        st.pyplot(fig)
    
    with chart_col3:
        #Plot Top 10 Carriers bar chart
        st.subheader("Top 10 Carriers with Bird Strikes")
        fig, ax = plt.subplots()
        ax.barh(carriers_df["Carrier"], carriers_df["Bird Strikes"], color='orange')
        ax.set_xlabel("Carrier")
        ax.set_ylabel("Number of Bird Strikes")
        ax.invert_yaxis()
        st.pyplot(fig)
       
###################################################################################################
#Configure Heatmap page.    
elif selected == "Heatmap":
    #Query to get coordinates data, with user option to filter year and operator.
    base_query = """
        SELECT 
            AIRPORT_LATITUDE, AIRPORT_LONGITUDE, INCIDENT_YEAR, TIME_OF_DAY, OPERATOR, COUNT(*) AS BIRD_STRIKES
        FROM FAA_BIRD_STRIKES
    """

    #Sidebar filters
    #Add year range filter
    year_range = st.sidebar.slider("Select Year Range", min_value=1996, max_value=2025, value=(1996, 2025), step=1)
    start_year, end_year = year_range
    
    #Operators filter
    query_operators = "SELECT DISTINCT OPERATOR FROM FAA_BIRD_STRIKES ORDER BY OPERATOR"
    cursor.execute(query_operators)
    operators = cursor.fetchall()
    operator_list = ["All"] + [operator[0] for operator in operators]
    selected_operator = st.sidebar.selectbox("Select Operator", operator_list)
    
    #Time of Day filter
    query_time_of_day = "SELECT DISTINCT TIME_OF_DAY FROM FAA_BIRD_STRIKES ORDER BY TIME_OF_DAY"
    cursor.execute(query_time_of_day)
    time_of_day = cursor.fetchall()
    time_of_day_list = ["All"] + [ToD[0] if ToD[0] is not None else "Unknown" for ToD in time_of_day]
    selected_time_of_day = st.sidebar.selectbox("Select Time of Day", time_of_day_list)

    #Dynamic query
    filters = []
    params = []
  
    #Apply year range filter
    filters.append("INCIDENT_YEAR BETWEEN %s AND %s")
    params.extend([start_year, end_year])

    #Apply operator filter if selected
    if selected_operator != "All":
        filters.append("OPERATOR = %s")
        params.append(selected_operator)

    #Apply time of day filter if selected
    if selected_time_of_day != "All":
        filters.append("TIME_OF_DAY = %s")
        params.append(selected_time_of_day)

    # uild final query
    where_clause = "WHERE " + " AND ".join(filters)
    query = f"{base_query} {where_clause} GROUP BY AIRPORT_LATITUDE, AIRPORT_LONGITUDE, INCIDENT_YEAR, TIME_OF_DAY, OPERATOR ORDER BY INCIDENT_YEAR"

    #Execute the query
    cursor.execute(query, params)
    airport_coords = cursor.fetchall()
    
    #Create coordinate df, convert coords to numeric
    coords_df = pd.DataFrame(airport_coords, columns=["Latitude", "Longitude", "Year", "TimeOfDay", "Operator", "Bird Strikes"])
    coords_df["Latitude"] = pd.to_numeric(coords_df["Latitude"], errors="coerce")
    coords_df["Longitude"] = pd.to_numeric(coords_df["Longitude"], errors="coerce")
    coords_df.dropna(subset=["Latitude", "Longitude"], inplace=True)

    global_map = folium.Map(location=[37.0902, -95.7129], zoom_start=2)
    heat_data = coords_df[["Latitude", "Longitude"]].values.tolist()
    HeatMap(heat_data).add_to(global_map)
    
    #Display map in app
    st.title(f"Heatmap of Global Bird Strikes ({start_year}-{end_year}) for Operator: {selected_operator}")
    folium_static(global_map)
    
###################################################################################################
#Configure Yearly Strikes page. 
elif selected == "Yearly Strikes":
    st.title("Yearly Number of Bird Strikes")
    
    #Query to get bird strikes grouped by year, with user option to filter
    base_query = """
        SELECT 
            INCIDENT_YEAR, TIME_OF_DAY, OPERATOR, AIRCRAFT, COUNT(*) AS BIRD_STRIKES
        FROM FAA_BIRD_STRIKES
    """
    
    #Sidebar filters
    #Operators filter
    query_operators = "SELECT DISTINCT OPERATOR FROM FAA_BIRD_STRIKES ORDER BY OPERATOR"
    cursor.execute(query_operators)
    operators = cursor.fetchall()
    operator_list = ["All"] + [ToD[0] if ToD[0] is not None else "Unknown" for ToD in operators]
    selected_operator = st.sidebar.selectbox("Select Operator", operator_list)
    
    #Time of Day filter
    query_time_of_day = "SELECT DISTINCT TIME_OF_DAY FROM FAA_BIRD_STRIKES ORDER BY TIME_OF_DAY"
    cursor.execute(query_time_of_day)
    time_of_day = cursor.fetchall()
    time_of_day_list = ["All"] + [ToD[0] if ToD[0] is not None else "Unknown" for ToD in time_of_day]
    selected_time_of_day = st.sidebar.selectbox("Select Time of Day", time_of_day_list)

    #Aircraft filter
    query_aircraft = "SELECT DISTINCT AIRCRAFT FROM FAA_BIRD_STRIKES ORDER BY AIRCRAFT"
    cursor.execute(query_aircraft)
    aircraft = cursor.fetchall()
    aircraft_list = ["All"] + [ToD[0] if ToD[0] is not None else "Unknown" for ToD in aircraft]
    selected_aircraft = st.sidebar.selectbox("Select Aircraft", aircraft_list)

    #Dynamic query
    filters = []
    params = []

    if selected_operator != "All":
        filters.append("OPERATOR = %s")
        params.append(selected_operator)
    if selected_time_of_day != "All":
        filters.append("TIME_OF_DAY = %s")
        params.append(selected_time_of_day)
    if selected_aircraft != "All":
        filters.append("AIRCRAFT = %s")
        params.append(selected_aircraft)

    if filters:
        where_clause = "WHERE " + " AND ".join(filters)
        query = f"{base_query} {where_clause} GROUP BY INCIDENT_YEAR, TIME_OF_DAY, OPERATOR, AIRCRAFT ORDER BY INCIDENT_YEAR"
    else:
        query = f"{base_query} GROUP BY INCIDENT_YEAR, TIME_OF_DAY, OPERATOR, AIRCRAFT ORDER BY INCIDENT_YEAR"

    #Execute the query
    cursor.execute(query, params)
    yearly_strikes = cursor.fetchall()
    
    #Create yearly strikes df
    yearly_strikes_df = pd.DataFrame(yearly_strikes, columns=["Year", "Time of Day", "Operator", "Aircraft", "Bird Strikes"])
    yearly_strikes_df["Year"] = pd.to_numeric(yearly_strikes_df["Year"], errors="coerce")
    yearly_strikes_df["Bird Strikes"] = pd.to_numeric(yearly_strikes_df["Bird Strikes"], errors="coerce")
 
    #Plot the data
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(yearly_strikes_df["Year"], yearly_strikes_df["Bird Strikes"], color="mediumpurple")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Bird Strikes", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
        
###################################################################################################
#Configure Time of Day page.
elif selected == "Time of Day":
    st.title("Number of Bird Strikes")
    #Create two columns for the bar charts side by side
    chart_col1, chart_col2 = st.columns(2)       
    with chart_col1:
        st.subheader("by Time of Day")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            time_of_day_df["Bird Strikes"], 
            labels=time_of_day_df["Time of Day"], 
            autopct="%1.1f%%", 
            startangle=90, 
            colors=plt.cm.tab10.colors
        )
        labeldistance=1.1,
        pctdistance=0.85 
        st.pyplot(fig)
    with chart_col2:
       st.subheader("by Hour")
       fig, ax = plt.subplots(figsize=(8, 8))
       ax.bar(time_df["Hour"], time_df["Bird Strikes"], color="aquamarine")
       ax.set_xlabel("Hour", fontsize=12)
       ax.set_xticks(time_df["Hour"]) 
       ax.tick_params(axis="x", rotation=0)
       ax.set_ylabel("Number of Bird Strikes", fontsize=12)
       ax.tick_params(axis="x", rotation=45)
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