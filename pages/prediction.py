import streamlit as st
import pandas as pd
import random
import plotly.graph_objects as go
import joblib
from assets.constants import WEATHER_FEATURES

st.title(':material/psychology: Prediction')
st.divider()

# Load models
models_dict = {
    'AdaBoost Regressor': {
        'model': joblib.load('models/ab_tune.pkl'),
        'pred': [],
        'series_pred': [],
    },
    'Random Forest Regressor': {
        'model': joblib.load('models/rf_tune.pkl'),
        'pred': [],
        'series_pred': [],
    },
    'XGBoost Regressor': {
        'model': joblib.load('models/xgb_tune.pkl'),
        'pred': [],
        'series_pred': [],
    },
}

# Load scaler
scaler = joblib.load('models/scaler.pkl')

# Load data
crop_data = pd.read_csv('data/crop_data_nasa.csv')
crop_data = crop_data[crop_data['Item'] == 'Rice'].reset_index(drop=True).drop(columns='Item')

# Layout
single_point_tab, multi_point_tab = st.tabs(['Model Playground', 'Yield Forecast based on Weather'])

with single_point_tab:

    # Input data
    X = {key: None for key in WEATHER_FEATURES.values()}

    with st.expander('Prediction Input', expanded=True):

        if st.button('Randomize', icon=':material/shuffle:', use_container_width=True):
            X['T2M_MIN'] = random.uniform(10, 50)
            X['T2M_MAX'] = random.uniform(10, 50)
            X['CLOUD_AMT'] = random.uniform(25, 100)
            X['WS10M'] = random.uniform(0, 10)
            X['PRECTOTCORR'] = random.uniform(0, 50)
            X['rain_days'] = random.randint(0, 365)
            X['GWETPROF'] = random.uniform(0, 100)
            X['GWETROOT'] = random.uniform(0, 100)
            X['GWETTOP'] = random.uniform(0, 100)

        inp_col_1, inp_col_2, inp_col_3, inp_col_4 = st.columns(4)

        with inp_col_1:
            st.markdown('**Temperature Data**')
            min_value, max_value = X['T2M_MIN'] if X['T2M_MIN'] is not None else 10.0,  X['T2M_MAX'] if X['T2M_MAX'] is not None else 50.0
            X['T2M_MIN'], X['T2M_MAX'] = st.slider(min_value=10.0, max_value=50.0, step=0.01, value=(min_value, max_value), label='Temperature (C)')
            X['T2M_RANGE'] = X['T2M_MAX'] - X['T2M_MIN']
        with inp_col_2:
            st.markdown('**Atmosphere Data**')
            X['CLOUD_AMT'] = st.slider(min_value=25.0, max_value=100.0, step=0.01, value=X['CLOUD_AMT'], label='Cloud Amount (%)')
            X['WS10M'] = st.slider(min_value=0.0, max_value=10.0, step=0.01, value=X['WS10M'], label='Wind Speed at 10 Meters (m/s)')
        with inp_col_3:
            st.markdown('**Rain Data**')
            X['PRECTOTCORR'] = st.slider(min_value=0.0, max_value=50.0, step=0.01, value=X['PRECTOTCORR'], label='Average Rainfall (mm/day)')
            X['rain_days'] = st.slider(min_value=0, max_value=365, step=1, value=X['rain_days'], label='Rainfall Days')
        with inp_col_4:
            st.markdown('**Soil Data**')
            X['GWETPROF'] = st.slider(min_value=0.0, max_value=100.0, step=0.01, value=X['GWETPROF'], label='Profile Soil Wetness (%)') / 100
            X['GWETROOT'] = st.slider(min_value=0.0, max_value=100.0, step=0.01, value=X['GWETROOT'], label='Root Zone Soil Wetness (%)') / 100
            X['GWETTOP'] = st.slider(min_value=0.0, max_value=100.0, step=0.01, value=X['GWETTOP'], label='Surface Soil Wetness (%)') / 100

    # Output
    out_col1, out_col2 = st.columns([1, 2])
    X_scaled_np = scaler.transform(pd.DataFrame([X], index=[0]))
    X_scaled = pd.DataFrame(X_scaled_np, columns=X.keys())

    # Predicted results from each model
    for model_name in models_dict.keys():
        models_dict[model_name]['pred'] = models_dict[model_name]['model'].predict(X_scaled)[0]

    @st.dialog(title='Model Summary', width='large')
    def show_model_summary(model, name):

        st.write(name)
        st.bar_chart( 
            pd.DataFrame(model.feature_importances_, index=WEATHER_FEATURES.keys()).reset_index(), 
            x='index', color='index', x_label='Weight / Importance', y_label='Feature', 
            height=500, horizontal=True)

        st.info('No idea what **Feature Importance** is? [Check out the link here!](https://www.baeldung.com/cs/ml-feature-importance#3-what-is-feature-importance-in-machine-learning)', icon=':material/help:')

    # Display the results as metrics
    with out_col1:

        average_yield = crop_data['Value'].mean()

        pred_average = sum([models_dict[name]['pred'] for name in models_dict.keys()]) / 3
        pct_chg_avg = (pred_average - average_yield) / average_yield * 100

        st.metric(label='Averaged Prediction Yield', value=f'{pred_average:.2f}', delta=f'{pct_chg_avg:.2f}% from historical average', border=True)

        show_individual_models_pred = st.toggle('View Individual Model Predictions')

        if show_individual_models_pred:

            for model_name in models_dict.keys():

                pct_chg = (models_dict[model_name]['pred'] - average_yield) / average_yield * 100

                st.metric(label=f'{model_name} Prediction Yield', value=f'{models_dict[model_name]['pred']:.2f}', delta=f'{pct_chg:.2f}% from historical average', border=True)

                if st.button(f'View {model_name} Model Summary', icon=':material/bar_chart:', use_container_width=True): 
                    show_model_summary(models_dict[model_name]['model'], model_name)

    with out_col2:

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=crop_data['Year'], y=crop_data['Value'], name='Historical Yield', line_shape='spline'))
        fig.add_trace(go.Scatter(
            x=crop_data['Year'], 
            y=[pred_average] * len(crop_data['Year']),
            mode='lines', 
            name='Average Predicted Yield',
            line=dict(dash='dash')
        ))

        if show_individual_models_pred:
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

with multi_point_tab:

    arima_pred = pd.read_csv('data/arima_pred.csv')
    arima_pred = arima_pred[arima_pred['YEAR'] > 2022]
    arima_pred['T2M_MAX'] = arima_pred['T2M_MIN'] + arima_pred['T2M_RANGE']
    arima_pred['rain_days'] = arima_pred['rain_days'].apply(int)

    with st.expander('View Forecasted Weather Data'):

        arima_pred_viz = arima_pred.copy()
        arima_pred_viz.columns = ['Year'] + list(WEATHER_FEATURES.keys())
        
        for col in ['Profile Soil Moisture (%)', 'Root Zone Soil Wetness (%)', 'Surface Soil Wetness (%)']:
            arima_pred_viz[col] *= 100

        arima_pred_viz = arima_pred_viz.set_index('Year').T
        arima_pred_viz['Trends'] = arima_pred_viz.apply(lambda row: row.tolist(), axis=1)

        st.dataframe(arima_pred_viz, 
                     column_config={ 'Trends': st.column_config.AreaChartColumn('Trend Line') },
                     use_container_width=True)

    arima_X = arima_pred.drop(columns='YEAR')
    arima_X_scaled = scaler.transform(arima_X)

    # Predicted results from each model
    for model_name in models_dict.keys():
        models_dict[model_name]['series_pred'] = models_dict[model_name]['model'].predict(arima_X_scaled)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=crop_data['Year'], y=crop_data['Value'], name='Historical Yield', line_shape='spline'))

    for model_name in models_dict.keys():
        fig.add_trace(go.Scatter(
            x=[2022] + list(arima_pred['YEAR']), 
            y=[crop_data[crop_data['Year'] == 2022]['Value'].values[0]] + list(models_dict[model_name]['series_pred']),
            mode='lines', 
            name=f'{model_name} Forecasted Yield',
            line_shape='spline'
        ))

    # Add title and labels
    fig.update_layout(
        title='Rice: Predicted Yield based on Weather Forecast',
        xaxis_title='Year',
        yaxis_title='Yield (kg/ha)',
        height=500
    )

    st.plotly_chart(fig)