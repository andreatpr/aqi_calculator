import joblib
import pandas as pd
import numpy as np
import streamlit as st
import requests
from bs4 import BeautifulSoup

scaler = joblib.load('./pkls/scaler_arreglado.pkl')
target_scaler = joblib.load('./pkls/scaler_target_arreglado.pkl')

class Data:
    def __init__(self, co, no2, o3, pm10, pm2_5, so2,
                 month, holiday, is_daytime, traffic_peak, distance_to_center):
        self.so2 = so2
        self.no2 = no2
        self.co = co
        self.o3 = o3
        self.pm10 = pm10
        self.pm2_5 = pm2_5
        self.month = month
        self.holiday = holiday
        self.is_daytime = is_daytime
        self.traffic_peak = traffic_peak
        self.distance_to_center = distance_to_center
        pass

def create_data(co, no2, o3, pm10, pm2_5, so2,
                month, holiday, is_daytime, traffic_peak, distance_to_center):
    
    dt = Data(co, no2, o3, pm10, pm2_5, so2,
                month, holiday, is_daytime, traffic_peak, distance_to_center)
    
    df = pd.DataFrame(dt.__dict__, index=[0])
    df.rename(columns={'pm2_5': 'pm2.5'}, inplace=True)
    return df

def predict(data):
    print(data)
    model = joblib.load('./pkls/modelo_optimizado_2.pkl')
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(data.mean(), inplace=True)
    timestamprn = pd.Timestamp.now()
    data_scaled = scaler.transform(data)
    aqi_predicted_scaled = model.predict(data_scaled)
    print(aqi_predicted_scaled)
    aqi_predicted = target_scaler.inverse_transform(np.array(aqi_predicted_scaled[0]).reshape(-1, 1))[0][0]
    timestamplt = timestamprn + 14*pd.Timedelta(hours=1)
    return aqi_predicted, timestamplt

def timerss ():
    timestamprn = pd.Timestamp.now() + 14 * pd.Timedelta(hours=1)
    month = timestamprn.month
    hour = timestamprn.hour
    is_daytime = 6 <= hour <= 18
    traffic_peak = 7 <= hour <= 9 or 17 <= hour <= 19

    holidays = {
        '01-01', '02-09', '02-10', '02-11', '03-01', '04-10', '05-05', 
        '05-15', '06-06', '08-15', '09-16', '09-17', '09-18', '10-01', 
        '10-03', '10-09', '12-24', '12-25', '12-31'
    }
    holiday = timestamprn.strftime('%m-%d') in holidays
    return month, holiday, is_daytime, traffic_peak

def get_air_quality_data(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code}")
    
    soup = BeautifulSoup(response.content, 'html.parser')

    pollutants = {
        'PM2.5': 1, 
        'PM10': 1, 
        'SO2': 1000, 
        'NO2': 1000, 
        'O3': 1000, 
        'CO': 1000
    }

    values = {}
    for pollutant, divisor in pollutants.items():
        element = soup.find('a', {'aria-label': f'View {pollutant} Page for this location'})
        value = element.find('span', {'title': True}).text.strip()
        values[pollutant] = float(value) / divisor

    month, holiday, is_daytime, traffic_peak = timerss()

    return (
        values['CO'], values['NO2'], values['O3'], 
        values['PM10'], values['PM2.5'], values['SO2'], 
        month, holiday, is_daytime, traffic_peak
    )

def predict_aqi(district):
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(district["url"])
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, district["distance"])
        aqi_predicted, timestamp = predict(data)
        st.markdown(f"### 🌟 Predicted AQI for **{district['name']}**:")
        st.write(f'{aqi_predicted} at {timestamp}')
        st.write("⚠️ only a prediction, actual AQI may vary.")
        st.write(f"🌐 [More information about {district['name']}]({district['url']})")


def main():
    st.title("🌎 Air Quality Index Prediction")
    st.markdown("""
        Welcome to the **Air Quality Index (AQI) Prediction** app for districts in Seoul!  
        Select a district to predict its AQI and view real-time insights.  
        🌬️ **Stay informed. Stay healthy.**
    """)
    districts = [
        {"name": "Jongno", "url": "https://www.aqi.in/dashboard/south-korea/seoul/seoul/jongno-gu", "distance": 2.271940},
        {"name": "Jung", "url": "https://www.aqi.in/dashboard/south-korea/seoul/seoul/jung-gu", "distance": 0.514527},
        {"name": "Yongsan", "url": "https://www.aqi.in/dashboard/south-korea/seoul/seoul/yongsan-gu", "distance": 3.658519},
        {"name": "Mapo", "url": "https://www.aqi.in/dashboard/south-korea/seoul/seoul/mapo-gu", "distance": 6.688899},
        {"name": "Dongdaemun-gu", "url": "https://www.aqi.in/dashboard/south-korea/seoul/seoul/dongdaemun-gu", "distance": 4.396950},
        {"name": "Gangbuk-gu", "url": "https://www.aqi.in/dashboard/south-korea/seoul/seoul/gangbuk-gu", "distance": 9.471084},
        {"name": "Dobong-gu", "url": "https://www.aqi.in/dashboard/south-korea/seoul/seoul/dobong-gu", "distance": 10.651130},
        {"name": "Nowon-gu", "url": "https://www.aqi.in/dashboard/south-korea/gyeonggi/guri/nowon-gu", "distance": 12.876435},
        {"name": "Yangcheon-gu", "url": "https://www.aqi.in/dashboard/south-korea/gyeonggi/gwangmyeongni/yangcheon-gu", "distance": 11.758872},
        {"name": "Gangseo-gu", "url": "https://www.aqi.in/dashboard/south-korea/gyeonggi/bucheon/gangseo-gu", "distance": 13.014215},
        {"name": "Guro-gu", "url": "https://www.aqi.in/dashboard/south-korea/gyeonggi/gwangmyeongni/guro-gu", "distance": 10.983758},
        {"name": "Geumcheon-gu", "url": "https://www.aqi.in/dashboard/south-korea/gyeonggi/gwangmyeongni/geumcheon-gu", "distance": 14.188581},
        {"name": "Dongjak-gu", "url": "https://www.aqi.in/dashboard/south-korea/gyeonggi/gwacheon/dongjak-gu", "distance": 9.544857},
        {"name": "Seocho-gu", "url": "https://www.aqi.in/dashboard/south-korea/gyeonggi/gwacheon/seocho-gu", "distance": 7.003440},
        {"name": "Gangnam-gu", "url": "https://www.aqi.in/dashboard/south-korea/gyeonggi/gwacheon/gangnam-gu", "distance": 8.083202},
        {"name": "Songpa-gu", "url": "https://www.aqi.in/dashboard/south-korea/gyeonggi/hanam/songpa-gu", "distance": 12.183902},
        {"name": "Gangdong-gu", "url": "https://www.aqi.in/dashboard/south-korea/gyeonggi/guri/gangdong-gu", "distance": 14.010065},
        {"name": "Seongbuk-gu", "url": "https://www.aqi.in/dashboard/south-korea/seoul/seoul/seongbuk-gu", "distance": 6.109974},
        {"name": "Gwanak-gu", "url": "https://www.aqi.in/dashboard/south-korea/gyeonggi/gwangmyeongni/gwanak-gu", "distance": 10.122347},
        {"name": "Jungnang-gu", "url": "https://www.aqi.in/dashboard/south-korea/gyeonggi/guri/jungnang-gu", "distance": 10.234600},
        {"name": "Yeongdeungpo-gu", "url": "https://www.aqi.in/dashboard/south-korea/gyeonggi/gwangmyeongni/yeongdeungpo-gu", "distance": 8.642297},
        {"name": "Gwangjin-gu", "url": "https://www.aqi.in/dashboard/south-korea/gyeonggi/guri/gwangjin-gu", "distance": 10.129872},
        {"name": "Seongdong-gu", "url": "https://www.aqi.in/dashboard/south-korea/seoul/seoul/seongdong-gu", "distance": 6.702829},
        {"name": "Seodaemun", "url": "https://www.aqi.in/dashboard/south-korea/seoul/seoul/seodaemun-gu", "distance": 4.045101}
    ]

    st.markdown("### 🏙️ Select a district to predict AQI:")
    district_names = [d["name"] for d in districts]
    selected_district = st.selectbox("Choose a district:", options=district_names)

    if st.button("🔍 Predict AQI"):
        district = next(d for d in districts if d["name"] == selected_district)
        predict_aqi(district)
    
if __name__ == "__main__":
    main()