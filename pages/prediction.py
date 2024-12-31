import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from assets.constants import WEATHER_FEATURES

st.title(':material/psychology: Prediction')
st.divider()

# Load models
models_dict = {
    'AdaBoost Regressor': {
        'model': joblib.load('models/ab_tune.pkl'),
        'pred': []
    },
    'Random Forest Regressor': {
        'model': joblib.load('models/rf_tune.pkl'),
        'pred': []
    },
    'XGBoost Regressor': {
        'model': joblib.load('models/xgb_tune.pkl'),
        'pred': []
    },
}

# Load scaler
scaler = joblib.load('models/scaler.pkl')

# Layout
single_point_tab, multi_point_tab = st.tabs(['Single Datapoint', 'Multi Datapoint'])

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
    # prediction_dict['AdaBoost Regressor'] = ab_reg.predict(X_scaled)[0]
    # prediction_dict['Random Forest Regressor'] = rf_reg.predict(X_scaled)[0]
    # prediction_dict['XGBoost Regressor'] = xgb_reg.predict(X_scaled)[0]
    for model_name in models_dict.keys():
        models_dict[model_name]['pred'] = models_dict[model_name]['model'].predict(X_scaled)[0]

    @st.dialog(title='Model Summary', width='large')
    def show_model_summary(model):
        st.bar_chart(
            # pd.DataFrame(model.feature_importances_, index=list(map(lambda x: WEATHER_FEATURES[x], X.keys()))).reset_index(), 
            pd.DataFrame(model.feature_importances_, index=WEATHER_FEATURES.keys()).reset_index(), 
            x='index', color='index', x_label='Weight', y_label='Feature', 
            height=500, horizontal=True)

    # Display the results as metrics
    with out_col1:

        for model_name in models_dict.keys():
            st.metric(label=f'{model_name} Prediction Yield', value=f'{models_dict[model_name]['pred']:.2f}', border=True)
            if st.button(f'View {model_name} Model Summary', icon=':material/bar_chart:', use_container_width=True): 
                show_model_summary(models_dict[model_name]['model'])

        # st.metric(label='AdaBoost Prediction Yield', value=f'{ab_pred:.2f}', border=True)
        # if st.button('View AdaBoost Regressor Model Summary', icon=':material/bar_chart:', use_container_width=True): 
        #     show_model_summary(ab_reg)

        # st.metric(label='Random Forest Prediction Yield', value=f'{rf_pred:.2f}', border=True)
        # if st.button('View Random Forest Regressor Model Summary', icon=':material/bar_chart:', use_container_width=True): 
        #     show_model_summary(rf_reg)

        # st.metric(label='XGBoost Prediction Yield', value=f'{xgb_pred:.2f}', border=True)
        # if st.button('View XGBoost Regressor Model Summary', icon=':material/bar_chart:', use_container_width=True): 
        #     show_model_summary(xgb_reg)

    with out_col2:

        crop_data = pd.read_csv('data/crop_data_nasa.csv')
        crop_data = crop_data[crop_data['Item'] == 'Rice'].reset_index(drop=True).drop(columns='Item')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=crop_data['Year'], y=crop_data['Value'], name='Historical Yield'))

        for model_name in models_dict.keys():
            fig.add_trace(go.Scatter(
                x=crop_data['Year'], 
                y=[models_dict[model_name]['pred']] * len(crop_data['Year']),
                mode='lines', 
                name=f'{model_name} Predicted Yield', 
                line=dict(dash='dash')
            ))

        # Add title and labels
        fig.update_layout(
            title='Rice: Predicted Yield and Historical Yield',
            xaxis_title='Year',
            yaxis_title='Yield (kg/ha)',
            legend=dict(
                orientation='h',
                x=0.5, xanchor='center',
                y=-0.35, yanchor='bottom'
            ),
            height=500
        )

        st.plotly_chart(fig)