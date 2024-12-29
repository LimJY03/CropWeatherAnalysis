import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from constants import WEATHER_FEATURES
import plotly.graph_objects as go
 


st.title(':material/monitoring: Data Visualization')
st.divider()

# Tabs
crop_weather_tab, crop_tab, weather_tab = st.tabs(['Crop Yield & Weather', 'Crop Yield', 'Weather'])

# Crop
with crop_tab:

    crop_data = pd.read_csv('data/crop_data_nasa.csv')
    crop_data = crop_data[crop_data['Item'] == 'Rice'].reset_index(drop=True).drop(columns='Item')
    crop_data['Pct_chg'] = (crop_data['Value'].pct_change() * 100) #.fillna(0)

    crop_feature_selection = list(crop_data.columns.drop('Year'))

    crop_col1, crop_col2 = st.columns([1, 2])

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

    with weather_col1:
        st.write('Select Column to Visualize:')
        selected_weather_feature = st.selectbox(options=WEATHER_FEATURES.keys(), label='select-weather', label_visibility='collapsed')
        
        st.dataframe(weather_data[['YEAR'] + [WEATHER_FEATURES[selected_weather_feature]]], 
                    column_config={'YEAR': st.column_config.NumberColumn(format='%f')},
                    hide_index=True, use_container_width=True)
        
        st.caption('Data sourced from [NASA Prediction Of Worldwide Energy Resources](https://power.larc.nasa.gov/data-access-viewer/).')

    with weather_col2:
        st.plotly_chart(px.line(weather_data, x='YEAR', y=WEATHER_FEATURES[selected_weather_feature], title=f'{selected_weather_feature} Across Year'))

# Crop yield and weather data
with crop_weather_tab:

    X = weather_data.drop(columns='YEAR')
    y = crop_data['Value']

    combine_left, combine_right = st.columns([1, 2])

    with combine_left:

        crop_feature = st.selectbox('Crop Feature', crop_feature_selection)
        weather_feature = st.selectbox('Weather Feature', WEATHER_FEATURES.keys())
        
        fig, ax = plt.subplots()
        sns.heatmap(X.corrwith(y).values.reshape(-1, 1), annot=True, yticklabels=X.columns, ax=ax)
        ax.set_title('Correlation between Weather Features and Crop Yield')
        st.pyplot(fig)

    crop_weather_data = crop_data.merge(weather_data, left_on='Year', right_on='YEAR').drop(columns='YEAR')    
    melt = crop_weather_data.melt(id_vars='Year', value_vars=[crop_feature, WEATHER_FEATURES[weather_feature]])

    with combine_right:

        fig = go.Figure() 
        fig.add_trace(go.Scatter(x=crop_weather_data['Year'], y=crop_weather_data[crop_feature], name=crop_feature, yaxis='y'))        
        fig.add_trace(go.Scatter(x=crop_weather_data['Year'], y=crop_weather_data[WEATHER_FEATURES[weather_feature]], name=weather_feature, yaxis='y2'))
        fig.update_layout(   
            yaxis=dict(title=crop_feature),
            yaxis2=dict(title=weather_feature, overlaying='y', side='right'),
            title_text=f'{crop_feature} and {weather_feature} Across Year',
        )

        st.plotly_chart(fig)