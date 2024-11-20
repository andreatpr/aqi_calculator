import joblib
import pandas as pd
import numpy as np
import streamlit as st
import requests
from bs4 import BeautifulSoup

scaler = joblib.load('scaler_arreglado.pkl')
target_scaler = joblib.load('scaler_target_arreglado.pkl')

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
    model = joblib.load('modelo_optimizado_2.pkl')
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(data.mean(), inplace=True)
    timestamprn = pd.Timestamp.now()
    data_scaled = scaler.transform(data)
    aqi_predicted_scaled = model.predict(data_scaled)
    print(aqi_predicted_scaled)
    aqi_predicted = target_scaler.inverse_transform(np.array(aqi_predicted_scaled[0]).reshape(-1, 1))[0][0]
    timestamplt = timestamprn + pd.Timedelta(hours=1)
    return aqi_predicted, timestamplt

def get_air_quality_data(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code}")
    
    soup = BeautifulSoup(response.content, 'html.parser')

    pm25_element = soup.find('a', {'aria-label': 'View PM2.5 Page for this location'})
    pm25_value = pm25_element.find('span', {'title': True}).text.strip()
    pm25_value = float(pm25_value)

    pm10_element = soup.find('a', {'aria-label': 'View PM10 Page for this location'})
    pm10_value = pm10_element.find('span', {'title': True}).text.strip()
    pm10_value = float(pm10_value)

    so2_element = soup.find('a', {'aria-label': 'View SO2 Page for this location'})
    so2_value = so2_element.find('span', {'title': True}).text.strip()
    so2_value = float(so2_value)/1000

    no2_element = soup.find('a', {'aria-label': 'View NO2 Page for this location'})
    no2_value = no2_element.find('span', {'title': True}).text.strip()
    no2_value = float(no2_value)/1000

    o3_element = soup.find('a', {'aria-label': 'View O3 Page for this location'})
    o3_value = o3_element.find('span', {'title': True}).text.strip()
    o3_value = float(o3_value)/1000

    co_element = soup.find('a', {'aria-label': 'View CO Page for this location'})
    co_value = co_element.find('span', {'title': True}).text.strip()
    co_value = float(co_value)/1000

    timestamprn = pd.Timestamp.now()

    month = timestamprn.month
    hour = timestamprn.hour
    is_daytime = False
    traffic_peak = False

    if 6 <= hour <= 18:
        is_daytime = True

    if 7 <= hour <= 9 or 17 <= hour <= 19:
        traffic_peak = True

    holidays = [
        '01-01', 
        '02-09', '02-10', '02-11',
        '03-01',  
        '04-10',  
        '05-05', 
        '05-15',
        '06-06',
        '08-15', 
        '09-16', '09-17', '09-18', 
        '10-01',
        '10-03', 
        '10-09', 
        '12-24', '12-25',  
        '12-31' 
    ]

    holiday = False
    if timestamprn.strftime('%m-%d') in holidays:
        holiday = True

    return co_value, no2_value, o3_value, pm10_value, pm25_value, so2_value, month, holiday, is_daytime, traffic_peak


def main():
    st.title('Air Quality Index Prediction')
    st.write('This is a simple app to predict the Air Quality Index (AQI) in the city of jongno.')
    
    if st.button('Jongno'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/jongno-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 2.271940
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        a, b = predict(data)
        st.write(f'Predicted AQI: {a} at {b}')

    if st.button('Jung'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/jung-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 0.514527
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Yongsan'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/yongsan-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 3.658519
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Mapo'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/mapo-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 6.688899
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Dongdaemun-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/dongdaemun-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 4.396950
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Gangbuk-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/gangbuk-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 9.471084
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Dobong-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/dobong-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 10.651130
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Nowon-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/nowon-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 12.876435
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Yangcheon-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/yangcheon-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 11.758872
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Gangseo-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/gangseo-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 13.014215
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Guro-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/guro-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 10.983758
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Geumcheon-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/geumcheon-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 14.188581
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Dongjak-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/dongjak-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 9.544857
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Seocho-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/seocho-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 7.003440
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Gangnam-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/gangnam-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 8.083202
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Songpa-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/songpa-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 12.183902
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Gangdong-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/gangdong-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 14.010065
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Eunpyeong-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/eunpyeong-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 6.274699
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Seongbuk-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/seongbuk-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 6.109974
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Gwanak-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/gwanak-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 9.967842
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Jungnang-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/jungnang-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 10.234600
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Yeongdeungpo-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/yeongdeungpo-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 8.642297
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Gwangjin-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/gwangjin-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 10.129872
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

    if st.button('Seongdong-gu'):
        url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/seongdong-gu'
        co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak = get_air_quality_data(url)
        distance_to_center = 6.702829
        data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
        aqi_predicted, timestamplt = predict(data)
        st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')




        if st.button('Seodaemun'):
            url = 'https://www.aqi.in/dashboard/south-korea/seoul/seoul/seodaemun-gu'
            co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak =get_air_quality_data(url)
            distance_to_center = 4.045101
            data = create_data(co, no2, o3, pm10, pm2_5, so2, month, holiday, is_daytime, traffic_peak, distance_to_center)
            aqi_predicted, timestamplt = predict(data)
            st.write(f'Predicted AQI: {aqi_predicted} at {timestamplt}')

        


if __name__ == '__main__':
    main()