# 🏡 US Housing Price Predictor (2020–2025)

A full-stack machine learning web application that predicts **house rental prices** in the United States using a trained **Random Forest model**, historical housing data, and **state-wise CAGR extrapolation** for future values (2021–2025).  
The app features a real-time **interactive map**, **dynamic validation**, **animated price charts**, and intuitive UI for users to input property details—either by **coordinates** or **full address**.

---

## 🚀 Features

### 📌 **Backend (Flask + ML)**
- Loads compressed Random Forest model and encoders.
- Accepts structured JSON input for property characteristics.
- Validates all inputs (numeric fields, coordinates, required fields).
- Encodes categorical variables (target encoding + label encoders).
- Scales features and predicts **2020 base price**.
- Uses **CAGR from state-wise historical averages** to project values to 2025.
- Returns:
  - Predicted price (2020)
  - Predicted price (2025)
  - All predicted yearly values (2020–2025)
  - Historical averages for the state
  - Selected CAGR

### 🖥️ **Frontend (HTML/JS)**
- Clean and interactive UI.
- Switch between **address mode** and **latitude/longitude** mode.
- Dynamic field validation with real-time feedback.
- Auto-map updates with OpenStreetMap (Leaflet.js).
- Searchable dropdowns for State & City.
- Animated Plotly chart:
  - Historical prices
  - Predicted prices
  - Auto-generated annotations for 2020 & 2025
- Error notifications with auto-dismiss animations.

---

## 🧠 Machine Learning

### **Model**
- Random Forest Regressor (trained on Kaggle USA Housing Listings).
- Target variable: **2020 rental price**.
- Inputs include:
  - Size & rooms: sqft, beds, baths
  - Region & house type
  - Boolean amenities (parking, laundry, pets, etc.)
  - Geolocation (lat/long)
  - State & City encodings

### **Future Price Projection (2021–2025)**
- Uses **historical state averages (2015–2025)**.
- Computes **CAGR between 2020 → 2025** per state.
- Applies CAGR to predicted 2020 price.

---

## 📦 Folder Structure

```

project/
│
├── README.md
├── app.py
├── kaggle_rf.py
├── prediction.py
├── requirements.txt
├── LICENSE
├── .gitattributes
├── .gitignore
│
├── csvs/
│   ├── housing.csv
│   ├── housing_sample2k.csv
│   ├── housing_sample10k.csv
│   ├── housing_sample20k.csv
│   ├── housing_sample50k.csv
│   ├── Metro_zori_uc_sfrcondomfr_sm_month.csv
│   ├── state_historic.csv
│   └── us_cities_states_counties.csv
│
├── extracts/
│   ├── cities_extract.py
│   ├── hist_data_extract.py
│   └── sample_extract.py
│
├── pkls/
│   ├── label_encoders.pkl
│   ├── random_forest_model.pkl
│   ├── random_forest_model_compressed.pkl
│   ├── scaler.pkl
│   └── target_encoder.pkl
│
├── static/
│   ├── canvas.js
│   ├── icons8-house-price-53.png
│   ├── script.js
│   ├── style.css
│   └── us-cities.json
│
└── templates/
    └── index.html

````

---

## 🔧 Installation & Setup

### **1. Clone repository**
```bash
git clone https://github.com/manivardhan-k/Rental-Price-Estimator.git
cd project-folder
````

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run the Flask server**

```bash
python app.py
```

Backend runs at:

```
http://127.0.0.1:5000
```

### **4. Open the frontend**

Visit:

```
http://127.0.0.1:5000/
```

---

## 📡 API Endpoint

### `POST /predict`

Send JSON body:

```json
{
  "sqft": 1200,
  "beds": 3,
  "baths": 2,
  "type": "apartment",
  "state": "CA",
  "city": "Los Angeles",
  "lat": 34.05,
  "long": -118.24,
  "cats_allowed": 1,
  "dogs_allowed": 0,
  "smoking_allowed": 0,
  "wheelchair_access": 0,
  "comes_furnished": 0,
  "electric_vehicle_charge": 0,
  "has_laundry": 1,
  "has_parking": 1
}
```

Response example:

```json
{
  "success": true,
  "price_2020": 1501.69,
  "price_2025": 1816.17,
  "predicted_prices": {
    "2020": 1501.69,
    "2021": 1559.89,
    "2022": 1620.36,
    "2023": 1683.16,
    "2024": 1748.40,
    "2025": 1816.17
  },
  "cagr": 0.03876,
  "historical_prices": {
    "2015": 1466.90,
    "2016": 1582.01,
    "2017": 1672.84,
    "2018": 1763.97,
    "2019": 1841.38,
    "2020": 1903.64,
    "2021": 2050.74,
    "2022": 2197.60,
    "2023": 2215.29,
    "2024": 2275.46,
    "2025": 2302.30
  },
  "state": "CA"
}
```

---

## 🌎 Data Sources

* Historical price data: Zillow Research
* House listings dataset: Kaggle – USA Housing Listings
* US States & Cities dataset: Kaggle
* Map tiles: OpenStreetMap

---

## ⚠️ Disclaimer

Predicted values are **estimates only** and should **not** be used for investment, legal, or financial decision-making.

---

## 👨‍💻 Developer

**Mani Vardhan Kumpatla**
Portfolio: [https://manivardhan-k.github.io/Portfolio/](https://manivardhan-k.github.io/Portfolio/)

