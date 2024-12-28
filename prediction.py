import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from constants import WEATHER_FEATURES

st.title(':material/psychology: Prediction')
st.divider()

# Load models
rf_reg = joblib.load('models/rf_tune.pkl')
ab_reg = joblib.load('models/ab_tune.pkl')
xgb_reg = joblib.load('models/xgb_tune.pkl')

# Load scaler
scaler = joblib.load('models/scaler.pkl')

# Layout
single_point_tab, multi_point_tab = st.tabs(['Single Datapoint', 'Multi Datapoint'])

make_predict = False

with single_point_tab:

    # Input data
    X = {key: '' for key in WEATHER_FEATURES.values()}

    with st.form(key='pred_input'):

        inp_col_1, inp_col_2, inp_col_3, inp_col_4 = st.columns(4)

        with inp_col_1:
            st.write('Temperature Data')
            min_temp = st.slider(min_value=20.0, max_value=40.0, step=0.01, label='Minimum Temperature (C)')
            max_temp = st.slider(min_value=20.0, max_value=40.0, step=0.01, label='Maximum Temperature (C)')
            X['T2M_RANGE'] = max_temp - min_temp
        with inp_col_2:
            st.write('Wind Data')
            X['CLOUD_AMT'] = st.slider(min_value=25.0, max_value=100.0, step=0.01, label='Cloud Amount (%)')
            X['WS10M'] = st.slider(min_value=0.0, max_value=10.0, step=0.01, label='Wind Speed at 10 Meters (m/s)')
        with inp_col_3:
            st.write('Rain Data')
            X['PRECTOTCORR'] = st.slider(min_value=0.0, max_value=50.0, step=0.01, label='Rainfall (mm/day)')
            X['PREC_DAYS'] = st.slider(min_value=80, max_value=365, step=1, label='Rainfall Days')
        with inp_col_4:
            st.write('Soil Data')
            X['GWETROOT'] = st.slider(min_value=0.0, max_value=100.0, step=0.01, label='Root Zone Soil Wetness (%)') / 100
            X['GWETTOP'] = st.slider(min_value=0.0, max_value=100.0, step=0.01, label='Surface Soil Wetness (%)') / 100
        
        st.form_submit_button(label='Predict', use_container_width=True)

    # Output
    out_col1, out_col2 = st.columns([1, 2])
    X_scaled = scaler.transform(pd.DataFrame([X], index=[0]))

    # Predicted results from each model
    rf_pred = rf_reg.predict(X_scaled)[0]
    ab_pred = ab_reg.predict(X_scaled)[0]
    xgb_pred = xgb_reg.predict(X_scaled)[0]

    st.write(X_scaled)
    st.write(rf_reg.feature_importances_)

    # Display the results as metrics
    with out_col1:
        st.metric(label='AdaBoost Prediction Yield', value=f'{ab_pred:.2f}', border=True)
        st.metric(label='Random Forest Prediction Yield', value=f'{rf_pred:.2f}', border=True)
        st.metric(label='XGBoost Prediction Yield', value=f'{xgb_pred:.2f}', border=True)

    with out_col2:
        st.bar_chart(pd.DataFrame({ 'Model': ['Random Forest Regressor', 'AdaBoost Regressor', 'XGBoost Regressor'],'Yield (kg/ht)': [rf_pred, ab_pred, xgb_pred] }), 
                     x='Model', 
                     y='Yield (kg/ht)', 
                     height=440,
                     horizontal=True,
                     use_container_width=True)