import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from assets.constants import WEATHER_FEATURES, WEATHER_RANGES
import plotly.graph_objects as go

st.title(':material/monitoring: Data Visualization')
st.divider()

# Tabs
crop_weather_tab, crop_tab, weather_tab = st.tabs(['Crop Yield & Weather', 'Crop Yield', 'Weather'])

# Crop
with crop_tab:

    crop_data = pd.read_csv('data/crop_data_nasa.csv').rename(columns={'Value': 'Yield (kg/ha)'})
    crop_data['Percent Change (%)'] = (crop_data['Yield (kg/ha)'].pct_change() * 100)

    crop_feature_selection = list(crop_data.columns.drop(['Year', 'Item']))

    crop_col1, crop_col2 = st.columns([1, 2])

    with crop_col1:
        selected_crop = st.selectbox('Select Crop Type', crop_data['Item'].unique())
        crop_data_viz = crop_data[crop_data['Item'] == selected_crop].reset_index(drop=True).drop(columns='Item')
        selected_crop_feature = st.selectbox('Select Column to Visualize:', crop_feature_selection, key='crop-col')

        st.dataframe(crop_data_viz[['Year'] + [selected_crop_feature]], 
                    column_config={'Year': st.column_config.NumberColumn(format='%f')},
                    hide_index=True, use_container_width=True)
        
        st.caption('Data sourced from [Food and Agriculture Organization of the United Nations](https://www.fao.org/faostat/en/#data/QCL).')

    with crop_col2:
        st.plotly_chart(px.line(crop_data_viz, x='Year', y=selected_crop_feature, title=f'{selected_crop_feature} Across Year', height=500))

# Weather data
with weather_tab: 

    weather_data = pd.read_csv('data/nasa_data_year.csv')
    sg_rain_data = pd.read_csv('data/sg_rain_data.csv')
    weather_data = weather_data.merge(sg_rain_data, left_on='YEAR', right_on='year')

    rows_to_convert = [WEATHER_FEATURES[col] for col, range in WEATHER_RANGES.items() if range == [0, 1]]

    for col in rows_to_convert:
        weather_data[col] *= 100

    weather_col1, weather_col2 = st.columns([1, 2])

    with weather_col1:
        st.write('Select Column to Visualize:')
        selected_weather_feature = st.selectbox(options=WEATHER_FEATURES.keys(), label='select-weather', label_visibility='collapsed')
        
        st.dataframe(weather_data[['YEAR'] + [WEATHER_FEATURES[selected_weather_feature]]], 
                    column_config={'YEAR': st.column_config.NumberColumn(format='%f')},
                    hide_index=True, use_container_width=True)
        
        st.caption('Data sourced from [NASA Prediction Of Worldwide Energy Resources](https://power.larc.nasa.gov/data-access-viewer/).')

    with weather_col2:
        st.plotly_chart(px.line(weather_data, x='YEAR', y=WEATHER_FEATURES[selected_weather_feature], title=f'{selected_weather_feature} Across Year', height=500))

# Crop yield and weather data
with crop_weather_tab:

    X = weather_data.drop(columns=['YEAR', 'year', 'PREC_DAYS'])

    combine_left, combine_right = st.columns([1, 2])

    with combine_left:

        selected_crop = st.selectbox('Select Crop Type', crop_data['Item'].unique(), key='multi-viz')
        crop_data_viz = crop_data[crop_data['Item'] == selected_crop].reset_index(drop=True).drop(columns='Item')
        y = crop_data_viz['Yield (kg/ha)']
        crop_feature = st.selectbox('Crop Feature', crop_feature_selection)
        weather_feature = st.selectbox('Weather Feature', WEATHER_FEATURES.keys())
        
        fig, ax = plt.subplots()
        sns.heatmap(X.corrwith(y).values.reshape(-1, 1), annot=True, yticklabels=WEATHER_FEATURES.keys(), ax=ax)
        ax.set_title(f'Correlation between Weather Features and {selected_crop} Yield')
        st.pyplot(fig)

    crop_weather_data = crop_data_viz.merge(weather_data, left_on='Year', right_on='YEAR').drop(columns='YEAR')    

    with combine_right:

        fig = go.Figure() 
        fig.add_trace(go.Scatter(x=crop_weather_data['Year'], y=crop_weather_data[crop_feature], name=crop_feature, yaxis='y'))        
        fig.add_trace(go.Scatter(x=crop_weather_data['Year'], y=crop_weather_data[WEATHER_FEATURES[weather_feature]], name=weather_feature, yaxis='y2'))
        fig.update_layout(   
            yaxis=dict(title=crop_feature),
            yaxis2=dict(title=weather_feature, overlaying='y', side='right'),
            title_text=f'{crop_feature} vs {weather_feature} Across Year',
            legend=dict(
                orientation='h',
                x=0.5, xanchor='center',
                y=-0.35, yanchor='bottom'
            ),
            height=500
        )

        st.plotly_chart(fig)