WEATHER_FEATURES = {
    'Cloud Amount (%)': 'CLOUD_AMT',
    'Profile Soil Moisture (%)': 'GWETPROF',
    'Root Zone Soil Wetness (%)': 'GWETROOT',
    'Surface Soil Wetness (%)': 'GWETTOP',
    'Average Rainfall (mm/day)': 'PRECTOTCORR',
    'Temperature at 2 Meters Maximum (C)': 'T2M_MAX',
    'Temperature at 2 Meters Minimum (C)': 'T2M_MIN',
    'Temperature at 2 Meters Range (C)': 'T2M_RANGE',
    'Wind Speed at 10 Meters (m/s)': 'WS10M',
    # 'Rainfall Days': 'PREC_DAYS',
    'Rainfall Days': 'rain_days'
}

WEATHER_RANGES = {
    'Cloud Amount (%)': [25, 100],
    'Profile Soil Moisture (%)': [0, 1],
    'Root Zone Soil Wetness (%)': [0, 1],
    'Surface Soil Wetness (%)': [0, 1],
    'Average Rainfall (mm/day)': [0, 50],
    'Temperature at 2 Meters Maximum (C)': [20, 50],
    'Temperature at 2 Meters Minimum (C)': [10, 40],
    'Temperature at 2 Meters Range (C)': [1, 10],
    'Wind Speed at 10 Meters (m/s)': [0, 10],
    # 'Rainfall Days': [0, 365],
    'Rainfall Days': 'rain_days'
}