WEATHER_FEATURES = {
    'Cloud Amount (%)': 'CLOUD_AMT',
    'Root Zone Soil Wetness': 'GWETROOT',
    'Surface Soil Wetness': 'GWETTOP',
    'Rainfall (mm/day)': 'PRECTOTCORR',
    'Temperature at 2 Meters Range (C)': 'T2M_RANGE',
    'Wind Speed at 10 Meters (m/s)': 'WS10M',
    'Rainfall Days': 'PREC_DAYS'
}

WEATHER_RANGES = {
    'Cloud Amount (%)': [25, 100],
    'Root Zone Soil Wetness': [0, 1],
    'Surface Soil Wetness': [0, 1],
    'Rainfall (mm/day)': [0, 50],
    'Temperature at 2 Meters Range (C)': [1, 10],
    'Wind Speed at 10 Meters (m/s)': [0, 10],
    'Rainfall Days': [80, 365],
}