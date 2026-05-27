### Imports
import pandas as pd
import streamlit as st
import warnings as wr
wr.filterwarnings('ignore')

# Sklearn Tools
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

### Load Dataset
CropName = pd.read_csv("Crop_Recomand.csv")

### Encoding 
Encoder = LabelEncoder()
CropName['label'] = Encoder.fit_transform(CropName['label'])

### Scaling: Normalization
numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
Scaler = MinMaxScaler()
CropName[numeric_cols] = Scaler.fit_transform(CropName[numeric_cols])

### Train Test Split
X = CropName.drop('label', axis=1)  # All Input
Y = CropName['label']               # All Output

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


### Selecting Best Model
Model_RFC = RandomForestClassifier()
Model_RFC.fit(X_train, Y_train)
Y_pred = Model_RFC.predict(X_test)


### Create Local Function
def Predict_crop(N, P, K, temp, humd, ph, rain):
    # Creat Dataframe from input values
    Input_data =pd.DataFrame([[N, P, K, temp, humd, ph, rain]],
                             columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    
    # Scale the Data
    Input_scaled = Scaler.transform(Input_data)

    # Predict & Decode
    Predic_encoded = Model_RFC.predict(Input_scaled)
    Prediction = Encoder.inverse_transform(Predic_encoded)

    return Prediction[0]

### UI Layout New
st.title('🌽 Crop Recomandation System')

st.markdown('### Please Enter The Details Values')

N = st.number_input('Nitrogen (N)', min_value=0.0, value=50.0)
P = st.number_input('Phosphorous (P)', min_value=0.0, value=50.0)
K = st.number_input('Potassium (K)', min_value=0.0, value=50.0)
temp = st.number_input('Temperature (C)', min_value=10.0, value=25.0)
humd = st.number_input('Humidity (%)', min_value=0.0, value=80.0)
ph = st.number_input('Soil pH', min_value=0.0, max_value=14.0, value=6.5)
rain = st.number_input('Railfall (mm)', min_value=0.0, value=100.0)

# Predict Button
if st.button('Recomanded'):
    Crop = Predict_crop(N, P, K, temp, humd, ph, rain)
    st.success(f"Recomanded Crop : **{Crop.capitalize()}**")
