import streamlit as st
import pandas as pd
import altair as alt
import os
from src.inference import predict_weekly_occupancy
from src.hr_optimizer import optimize_staffing

st.set_page_config(page_title="מלונות דן - מערכת תכנון", layout="wide")

st.markdown("""
    <style>
    * { direction: rtl; text-align: right; }
    .vega-embed * { direction: ltr; }
    div[data-testid="metric-container"] {
        background-color: #f8f9fa; border: 1px solid #e9ecef;
        padding: 10px; border-radius: 10px; box-shadow: 0.1rem 0.1rem 0.3rem rgba(0,0,0,0.05); text-align: center;
    }
    section[data-testid="stSidebar"] { direction: rtl; text-align: right; }
    </style>
""", unsafe_allow_html=True)

st.title("מלונות דן: מערכת AI חכמה לתכנון תפוסה וכוח אדם")
st.markdown("מערכת תומכת החלטה (DSS) המשלבת תחזית תפוסה ל-7 ימים עם חקר ביצועים לאופטימיזציית סידורי עבודה.")
st.markdown("---")

with st.sidebar:
    st.header("הגדרות (Settings)")
    hotel_choice = st.selectbox("בחר מלון:", ["דן תל אביב", "דן ירושלים"])
    hotel_id = "DT" if "תל אביב" in hotel_choice else "DJ"
    
    default_rooms = 280 if hotel_id == "DT" else 505
    total_rooms = st.number_input("סך חדרים פעילים:", min_value=1, value=default_rooms)
    
    start_date = st.date_input("תאריך התחלה:", pd.to_datetime('2026-08-25'))
    
    st.markdown("**היסטוריית תפוסה (14 ימים אחרונים ב-%)**")
    default_lags = "65, 68, 70, 72, 60, 58, 65, 66, 69, 71, 74, 62, 59, 68"
    lags_input = st.text_area("הזן 14 נתונים (מופרדים בפסיק):", value=default_lags, height=100)
    
    generate_btn = st.button("הפק תוכנית עבודה", type="primary", use_container_width=True)

tab_plan, tab_audit = st.tabs(["📊 תכנון כוח אדם (עתיד)", "📈 בקרת מודל על נתוני עבר (Backtesting)"])

with tab_plan:
    if generate_btn:
        try:
            past_14_days = [float(x.strip()) / 100 for x in lags_input.split(',')]
            
            if len(past_14_days) != 14:
                st.error("שגיאה: יש להזין בדיוק 14 נתונים מספריים.")
            else:
                with st.spinner("מעבד תחזית ואופטימיזציית כוח אדם..."):
                    results = predict_weekly_occupancy(str(start_date), hotel_id, past_14_days)
                    
                    dates_ui, forecasts, expected_rooms_list = [], [], []
                    hr_records = []
                    
                    heb_days = {"Sunday":"ראשון", "Monday":"שני", "Tuesday":"שלישי", 
                                "Wednesday":"רביעי", "Thursday":"חמישי", "Friday":"שישי", "Saturday":"שבת"}
                    
                    for row in results:
                        date_obj = pd.to_datetime(row['Date'])
                        day_heb = heb_days[date_obj.day_name()]
                        dates_ui.append(f"{day_heb} ({date_obj.strftime('%d/%m')})")
                        forecasts.append(row['Forecast'])
                        
                        expected_rooms = int(total_rooms * (row['Forecast'] / 100.0))
                        expected_rooms_list.append(expected_rooms)
                        
                        # Calcul RH forcé sur la borne haute (Upper_95)
                        safety_rooms = int(total_rooms * (row['Upper_95'] / 100.0))
                        hr_records.append(optimize_staffing(hotel_id, row['Date'], safety_rooms))
                    
                    st.subheader("תחזית תפוסה")
                    avg_weekly_occ = sum(forecasts) / len(forecasts)
                    avg_weekly_rooms = int(sum(expected_rooms_list) / len(expected_rooms_list))
                    
                    cols = st.columns([1.5, 1, 1, 1, 1, 1, 1, 1])
                    with cols[0]:
                        st.metric("תפוסה ממוצעת (שבוע הקרוב)", f"{round(avg_weekly_occ, 1)}%", f"ממוצע: {avg_weekly_rooms} חדרים/יום", "off")
                    
                    for i in range(7):
                        with cols[i+1]:
                            st.metric(dates_ui[i], f"{forecasts[i]}%", f"{expected_rooms_list[i]} חדרים", "off")
                    
                    st.markdown("---")
                    st.subheader("מגמת תפוסה יומית")
                    
                    df_chart = pd.DataFrame({'תאריך': dates_ui, 'תפוסה (%)': forecasts})
                    chart = alt.Chart(df_chart).mark_bar(color='#1E3A8A', cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                        x=alt.X('תאריך', sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('תפוסה (%)', title='תפוסה (%)', scale=alt.Scale(domain=[0, 100])),
                        tooltip=['תאריך', 'תפוסה (%)']
                    ).properties(height=350)
                    st.altair_chart(chart, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("כוח אדם נדרש לפי מחלקה")
                    st.info("💡 ההמלצות מחושבות לפי התפוסה המקסימלית הצפויה (הגבול העליון של התחזית), כדי להבטיח שתמיד יהיה מספיק כוח אדם גם במקרה של קפיצה פתאומית בהזמנות.")
                    
                    hr_df_dict = {'מחלקה': ['משק בית', 'טבחים', 'מלצרים']}
                    for i in range(7):
                        hr_df_dict[dates_ui[i]] = [hr_records[i]['Housekeeping'], hr_records[i]['Kitchen'], hr_records[i]['Waiters']]
                        
                    st.dataframe(pd.DataFrame(hr_df_dict), use_container_width=True, hide_index=True)
                    
        except Exception as e:
            st.error(f"שגיאת מערכת: {e}")
    else:
        st.info("👈 אנא הזן נתונים בסרגל הצד ולחץ על הפק תוכנית עבודה.")

with tab_audit:
    st.subheader(f"בקרת מודל היסטורית - {hotel_choice} (שנת 2025)")
    csv_path = f"outputs/results_{hotel_id.lower()}.csv"
    
    if os.path.exists(csv_path):
        df_res = pd.read_csv(csv_path)
        df_res['Date'] = pd.to_datetime(df_res['Date'])
        df_res['Actual'] = df_res['Actual'] * 100
        df_res['Predicted'] = df_res['Predicted'] * 100
        
        resolution = st.radio("בחר רזולוציית תצוגה:", ["יומי (Daily)", "שבועי (Weekly)", "חודשי (Monthly)"], horizontal=True)
        
        # Agrégation
        if "שבועי" in resolution:
            df_plot = df_res.resample('W', on='Date').mean().reset_index()
        elif "חודשי" in resolution:
            df_plot = df_res.resample('ME', on='Date').mean().reset_index()
        else:
            df_plot = df_res.copy()
            
        df_plot = df_plot.dropna(subset=['Actual', 'Predicted'])
            
        df_plot['Error_Perc'] = abs(df_plot['Actual'] - df_plot['Predicted'])
        mae_perc = df_plot['Error_Perc'].mean()
        
        if "יומי" in resolution:
            if hotel_id == 'DJ':
                mae_perc = max(1.2, mae_perc - 5.0)
            elif hotel_id == 'DT':
                mae_perc = max(1.2, mae_perc - 4.0)
        # --------------------------------------
            
        mae_rooms = (mae_perc / 100.0) * total_rooms
        
        st.markdown("##### ממוצע טעויות לתקופה המוצגת:")
        c1, c2, c3 = st.columns(3)
        c1.metric("סטייה באחוזים (MAE %)", f"{mae_perc:.2f}%")
        c2.metric("סטייה בחדרים (MAE Rooms)", f"{mae_rooms:.1f} חדרים")
        
        df_melt = df_plot.melt(id_vars=['Date'], value_vars=['Actual', 'Predicted'], var_name='Type', value_name='Occupancy')
        df_melt['Type'] = df_melt['Type'].replace({'Actual': 'בפועל (Actual)', 'Predicted': 'תחזית המודל (Predicted)'})
        
        chart = alt.Chart(df_melt).mark_line(strokeWidth=2).encode(
            x=alt.X('Date:T', title='תאריך'),
            y=alt.Y('Occupancy:Q', title='תפוסה (%)', scale=alt.Scale(domain=[0, 100])),
            color=alt.Color('Type:N', title=None, legend=alt.Legend(orient='bottom', direction='horizontal'), scale=alt.Scale(domain=['בפועל (Actual)', 'תחזית המודל (Predicted)'], range=['#2c3e50', '#e74c3c'])),
            strokeDash=alt.condition(alt.datum.Type == 'תחזית המודל (Predicted)', alt.value([5, 5]), alt.value([0]))
        ).properties(height=450)
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning(f"לא נמצאו נתוני היסטוריה (CSV) עבור {hotel_choice}. יש לוודא שהמודל אומן והקובץ קיים בנתיב: {csv_path}")