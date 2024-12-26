import streamlit as st

# Config
st.set_page_config(layout='wide')

# Paging
dashboard_page = st.Page('dashboard.py', title='Dashboard', icon=':material/monitoring:')
prediction_page = st.Page('prediction.py', title='Prediction', icon=':material/psychology:')
manual_page = st.Page('manual.py', title='User Manual', icon=':material/menu_book:')
pg = st.navigation([dashboard_page, prediction_page, manual_page])
pg.run()

# Footer
st.divider()
st.write('Copyright :material/copyright: 2024 - 2025 Lim Jun Yi. All Rights Reserved.')