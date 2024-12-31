import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from assets.constants import WEATHER_FEATURES

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
            st.markdown('**Temperature Data**')
            # X['T2M_MAX'] = st.slider(min_value=20.0, max_value=50.0, step=0.01, label='Maximum Temperature (C)')
            # X['T2M_MIN'] = st.slider(min_value=10.0, max_value=40.0, step=0.01, label='Minimum Temperature (C)')
            X['T2M_MIN'], X['T2M_MAX'] = st.slider(min_value=10.0, max_value=50.0, step=0.01, value=(10.0, 50.0), label='Temperature (C)')
            # X['T2M_RANGE'] = st.text_input(value=X['T2M_MAX'] - X['T2M_MIN'], disabled=True, label='Temperature Range (C)')
            X['T2M_RANGE'] = X['T2M_MAX'] - X['T2M_MIN']
        with inp_col_2:
            st.markdown('**Wind Data**')
            X['CLOUD_AMT'] = st.slider(min_value=25.0, max_value=100.0, step=0.01, label='Cloud Amount (%)')
            X['WS10M'] = st.slider(min_value=0.0, max_value=10.0, step=0.01, label='Wind Speed at 10 Meters (m/s)')
        with inp_col_3:
            st.markdown('**Rain Data**')
            X['PRECTOTCORR'] = st.slider(min_value=0.0, max_value=50.0, step=0.01, label='Average Rainfall (mm/day)')
            X['PREC_DAYS'] = st.slider(min_value=0, max_value=365, step=1, label='Rainfall Days')
        with inp_col_4:
            st.markdown('**Soil Data**')
            X['GWETPROF'] = st.slider(min_value=0.0, max_value=100.0, step=0.01, label='Profile Soil Wetness (%)') / 100
            X['GWETROOT'] = st.slider(min_value=0.0, max_value=100.0, step=0.01, label='Root Zone Soil Wetness (%)') / 100
            X['GWETTOP'] = st.slider(min_value=0.0, max_value=100.0, step=0.01, label='Surface Soil Wetness (%)') / 100
        
        st.form_submit_button(label='Predict', type='primary', use_container_width=True)

    # Output
    out_col1, out_col2 = st.columns([1, 2])
    X_scaled_np = scaler.transform(pd.DataFrame([X], index=[0]))
    X_scaled = pd.DataFrame(X_scaled_np, columns=X.keys())

    # Predicted results from each model
    rf_pred = rf_reg.predict(X_scaled)[0]
    ab_pred = ab_reg.predict(X_scaled)[0]
    xgb_pred = xgb_reg.predict(X_scaled)[0]

    @st.dialog(title='Model Summary', width='large')
    def show_model_summary(model):
        st.bar_chart(
            # pd.DataFrame(model.feature_importances_, index=list(map(lambda x: WEATHER_FEATURES[x], X.keys()))).reset_index(), 
            pd.DataFrame(model.feature_importances_, index=WEATHER_FEATURES.keys()).reset_index(), 
            x='index', color='index', x_label='Weight', y_label='Feature', 
            height=500, horizontal=True)

    # Display the results as metrics
    with out_col1:
        st.metric(label='AdaBoost Prediction Yield', value=f'{ab_pred:.2f}', border=True)
        if st.button('View AdaBoost Regressor Model Summary', icon=':material/bar_chart:', use_container_width=True): 
            show_model_summary(ab_reg)

        st.metric(label='Random Forest Prediction Yield', value=f'{rf_pred:.2f}', border=True)
        if st.button('View Random Forest Regressor Model Summary', icon=':material/bar_chart:', use_container_width=True): 
            show_model_summary(rf_reg)

        st.metric(label='XGBoost Prediction Yield', value=f'{xgb_pred:.2f}', border=True)
        if st.button('View XGBoost Regressor Model Summary', icon=':material/bar_chart:', use_container_width=True): 
            show_model_summary(xgb_reg)

    with out_col2:
        st.write('')
        st.write('')
        st.markdown('**Model Predictions**')
        st.bar_chart(pd.DataFrame({ 'Model': ['Random Forest Regressor', 'AdaBoost Regressor', 'XGBoost Regressor'],'Yield (kg/ht)': [rf_pred, ab_pred, xgb_pred] }), 
                     x='Model', 
                     y='Yield (kg/ht)', 
                     color='Model',
                     height=440,
                     horizontal=True,
                     use_container_width=True)