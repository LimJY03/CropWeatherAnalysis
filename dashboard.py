import streamlit as st
import pandas as pd
import plotly.express as px
from constants import WEATHER_FEATURES

st.title(':material/monitoring: Data Visualization')
st.divider()

# Tabs
crop_tab, weather_tab = st.tabs(['Crop Yield', 'Weather'])

# Crop
with crop_tab:

    crop_data = pd.read_csv('data/crop_data_nasa.csv')
    crop_data = crop_data[crop_data['Item'] == 'Rice'].reset_index(drop=True).drop(columns='Item')
    crop_data['Pct_chg'] = (crop_data['Value'].pct_change() * 100) #.fillna(0)

    crop_feature_selection = list(crop_data.columns.drop('Year'))

    crop_col1, crop_col2 = st.columns([1, 2])
    selected_crop_feature = crop_feature_selection[0]

    with crop_col1:
        st.write('Select Column to Visualize:')
        selected_crop_feature = st.selectbox(options=crop_feature_selection, label='select-crop', label_visibility='collapsed')

        st.dataframe(crop_data[['Year'] + [selected_crop_feature]], 
                    column_config={'Year': st.column_config.NumberColumn(format='%f')},
                    hide_index=True, use_container_width=True)
        
        st.caption('Data sourced from [Food and Agriculture Organization of the United Nations](https://www.fao.org/faostat/en/#data/QCL).')

    with crop_col2:
        st.plotly_chart(px.line(crop_data, x='Year', y=selected_crop_feature, title=f'{selected_crop_feature} Across Year'))

# Weather data
with weather_tab: 

    weather_data = pd.read_csv('data/nasa_data_year.csv')

    weather_col1, weather_col2 = st.columns([1, 2])
    selected_weather_feature = list(WEATHER_FEATURES.keys())[0]

    with weather_col1:
        st.write('Select Column to Visualize:')
        selected_weather_feature = st.selectbox(options=WEATHER_FEATURES.keys(), label='select-weather', label_visibility='collapsed')
        
        st.dataframe(weather_data[['YEAR'] + [WEATHER_FEATURES[selected_weather_feature]]], 
                    column_config={'YEAR': st.column_config.NumberColumn(format='%f')},
                    hide_index=True, use_container_width=True)
        
        st.caption('Data sourced from [NASA Prediction Of Worldwide Energy Resources](https://power.larc.nasa.gov/data-access-viewer/).')

    with weather_col2:
        st.plotly_chart(px.line(weather_data, x='YEAR', y=WEATHER_FEATURES[selected_weather_feature], title=f'{selected_weather_feature} Across Year'))