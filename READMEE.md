# AI-Based Landslide & Flood Risk Monitoring System

## SIH 2026 – SIH26001

An AI-based early warning and risk monitoring system designed for landslide-prone regions of the **North Eastern Region (NER) of India**.

The system combines rainfall, weather, terrain, elevation, slope, and flood indicators to estimate landslide and overall hazard risk for a selected location.

---

## Problem Statement

**SIH26001 – AI-Based Early Warning and Landslide Risk Monitoring System in NER**

The objective of this project is to develop an AI-assisted monitoring system that can assess landslide and flood-related hazards using environmental and terrain information and provide an easy-to-understand risk assessment through an interactive dashboard.

---

## Key Features

- GPS-based location selection
- Manual latitude and longitude input
- Real-time weather and rainfall data
- Elevation and terrain analysis
- Slope calculation
- AI-based landslide risk prediction
- Flood-risk indicator using river discharge data
- Overall hazard classification
- Offline risk assessment using cached data
- Interactive web dashboard
- FastAPI backend
- Streamlit frontend
- Scalable architecture for multiple regions

---

## System Architecture

```text
                 LOCATION
              GPS / Manual
                    |
                    v
            +---------------+
            |    FastAPI    |
            |    Backend    |
            +-------+-------+
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
    WEATHER      TERRAIN      FLOOD
    RAINFALL     ELEVATION     RIVER
    HUMIDITY     SLOPE         DISCHARGE
    TEMPERATURE
        |           |           |
        +-----------+-----------+
                    |
                    v
          AI RISK ASSESSMENT
                    |
          +---------+---------+
          |                   |
          v                   v
   LANDSLIDE RISK        FLOOD RISK
          |                   |
          +---------+---------+
                    |
                    v
             OVERALL HAZARD
                    |
                    v
          STREAMLIT DASHBOARD
                    |
                    v
          LOCAL / OFFLINE
          RISK MONITORING


Methodology
1. Location Acquisition

The user can select a location using:

GPS
Manual latitude and longitude
Predefined example locations
2. Environmental Data Collection

The system collects:

Rainfall
Temperature
Humidity
Soil-related weather information
Recent rainfall information
3. Terrain Analysis

Elevation data is obtained and terrain information is processed to estimate:

Elevation
Slope
Aspect
4. AI-Based Risk Prediction

A Logistic Regression model is used to estimate landslide risk using:

Rainfall
Slope Angle
5. Flood Risk Assessment

River discharge information is obtained through the flood-data service and used as a flood-risk indicator.

6. Overall Hazard Assessment

The system combines landslide and flood assessments to determine an overall hazard level.

7. Offline Monitoring

Previously retrieved weather information is stored in a local cache.

When fresh weather information is unavailable, cached data can be used for local risk assessment.

Offline mode uses cached data and does not provide fresh real-time weather information.

8. Visualization

The final risk information is displayed through an interactive Streamlit dashboard.

AI Model

The live prediction model is a Logistic Regression model.

Model Features
Rainfall_mm
Slope_Angle

The model uses a StandardScaler before Logistic Regression.

Model Pipeline
Input Data
    |
    v
StandardScaler
    |
    v
Logistic Regression
    |
    v
Risk Score
    |
    v
Risk Level
Model Performance

The prototype model was evaluated using the WSN Landslide Dataset.

Results
Test Accuracy : 97.82%
ROC-AUC        : 0.9811

The confusion matrix on the test set was:

[[963, 18],
 [ 25, 967]]

These results are based on the prototype dataset and should not be interpreted as real-world prediction accuracy.

Risk Classification

The model output is converted into a risk score from 0 to 100.

Risk Score	Risk Level
0–24	LOW
25–49	MEDIUM
50–74	HIGH
75–100	VERY HIGH

The risk score is a prototype AI-based risk indicator.

It is not an official government warning level.

Data Sources
Open-Meteo

Used for:

Weather information
Rainfall
Temperature
Humidity
Soil-related weather information
Elevation
Copernicus DEM

Elevation information is obtained through the Open-Meteo elevation service based on Copernicus DEM data.

GloFAS

River discharge information is used as a flood-risk indicator through the Open-Meteo flood service.

WSN Landslide Dataset

Used as the primary dataset for training and evaluating the machine-learning model.

NASA Global Landslide Catalog

Used as a historical landslide inventory for reference and analysis.

UGLC

Used as a large-scale global landslide inventory for reference and testing.

Dataset

The primary machine-learning dataset is:

wsn_landslide_data.csv

Dataset information:

Rows    : 9,864
Columns : 35
Target  : Label

The target label represents:

0 = No Landslide
1 = Landslide
Technology Stack
Programming Language
Python
Machine Learning
Scikit-learn
Logistic Regression
StandardScaler
Joblib
Data Processing
Pandas
NumPy
Backend
FastAPI
Uvicorn
Frontend
Streamlit
Communication
REST API
JSON
HTTP Requests
External Data Services
Open-Meteo
Copernicus DEM
GloFAS
Project Structure
Landslides_monitoring/
│
├── app.py
├── dashboard.py
│
├── landslide_live_model.pkl
├── live_feature_names.json
├── weather_cache.json
│
├── requirements.txt
├── README.md
│
└── data/
    └── wsn_landslide_data.csv
Backend API

The backend is implemented using FastAPI.

Main API Endpoints
GET /

Returns system information and API status.

POST /location-features

Returns environmental and terrain features for a selected location.

POST /predict

Performs AI-based landslide risk prediction.

POST /complete-risk

Performs complete risk assessment including:

Weather
Terrain
Landslide risk
Flood indicator
Overall hazard
Local warning information
Installation
1. Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
2. Open the Project
cd Landslides_monitoring
3. Install Dependencies
pip install -r requirements.txt
Run the Backend

Open a terminal and run:

uvicorn app:app --host 127.0.0.1 --port 8000

The API will run at:

http://127.0.0.1:8000

You can check the API by opening:

http://127.0.0.1:8000/
Run the Dashboard

Open another terminal:

streamlit run dashboard.py

The dashboard will normally be available at:

http://localhost:8501
How the System Works
User selects location
        |
        v
Latitude & Longitude
        |
        v
FastAPI Backend
        |
        +------------------+
        |                  |
        v                  v
Weather Data          Elevation Data
        |                  |
        v                  v
Rainfall              Slope Calculation
        |                  |
        +--------+---------+
                 |
                 v
        Logistic Regression
                 |
                 v
        Landslide Risk Score
                 |
                 v
          Flood Indicator
                 |
                 v
         Overall Hazard
                 |
                 v
       Streamlit Dashboard
Offline Mode

The system includes a weather cache:

weather_cache.json

The cache stores previously retrieved weather information.

If fresh weather information cannot be obtained, the backend can use available cached information for risk assessment.

The system can therefore continue displaying a local risk assessment when the external weather service is temporarily unavailable.

Important Limitation

The current implementation is a web-based prototype.

It does not provide:

Fresh weather data without internet
Mobile-to-mobile offline communication
Bluetooth mesh alerts
SMS alerts
Push notifications to nearby people
Alerts

The current system generates local risk/warning information on the dashboard.

It does not send notifications to nearby people.

No user database, device-token database, Firebase Cloud Messaging system, or SMS gateway is currently required.

This keeps the current prototype focused on:

Risk Detection
       +
Risk Visualization
       +
Local / Offline Monitoring
NER-First Design

The project is designed primarily for the North Eastern Region (NER) of India, as required by the SIH problem statement.

The architecture is designed to be scalable so that additional regions can be supported in the future.

NER-FIRST
    |
    +---- Arunachal Pradesh
    |
    +---- Assam
    |
    +---- Manipur
    |
    +---- Meghalaya
    |
    +---- Mizoram
    |
    +---- Nagaland
    |
    +---- Sikkim
    |
    +---- Tripura
    |
    v
Future Multi-Region Expansion
Advantages
Location-based risk assessment
AI-assisted landslide prediction
Combines rainfall and terrain information
Includes flood-risk monitoring
Easy-to-use dashboard
Supports GPS and manual location input
Uses external environmental data
Provides offline cached-data fallback
Can be extended to additional regions
Separate backend and frontend architecture
Limitations

The current prototype has the following limitations:

The ML model is trained using a prototype dataset.
The reported accuracy is dataset-based and does not represent field accuracy.
Risk scores are indicators and not official warnings.
Offline operation depends on previously cached information.
Fresh environmental data requires internet connectivity.
The current weather cache is intended for prototype use.
Nearby-person notification is not implemented.
More real-world NER landslide observations are required for production deployment.
Future Enhancements

Future versions can include:

Real-world NER landslide datasets
Satellite imagery analysis
Remote sensing-based monitoring
Soil moisture analysis
NDVI and vegetation monitoring
Improved model calibration
Location-specific persistent offline cache
Mobile application / PWA
Stronger offline operation
Integration with official disaster-management systems
Historical landslide visualization
Time-series risk prediction
More advanced machine-learning models
Deployment

The system can be deployed using separate services:

                 Internet
                    |
          +---------+---------+
          |                   |
          v                   v
   Streamlit Cloud          Render
     Frontend              Backend
          |                   |
          +---------+---------+
                    |
                    v
             External APIs
Backend

The FastAPI backend can be deployed on a cloud service such as Render.

Frontend

The Streamlit dashboard can be deployed using Streamlit Community Cloud.

After deployment, the dashboard API URL should be updated from:

API_URL = "http://127.0.0.1:8000"

to the deployed FastAPI backend URL.

Security and Reliability

The prototype follows a modular architecture:

Frontend
   |
   v
Backend API
   |
   +---- ML Model
   |
   +---- Weather API
   |
   +---- Elevation API
   |
   +---- Flood API

This makes it easier to replace individual components and extend the system in future versions.

Project Objective

The main objective is to provide an accessible AI-assisted monitoring platform that can help identify potential landslide and flood hazards using environmental and terrain information.

The system aims to support early awareness and risk monitoring, particularly for vulnerable regions of the North Eastern Region of India.

Team
Team CodeForge

Smart India Hackathon 2026

Problem Statement: SIH26001

Theme: Disaster Management

Category: Software

Disclaimer

This project is an academic and Smart India Hackathon prototype.

The AI-generated risk score is intended for demonstration and risk-monitoring purposes only. It should not be treated as an official disaster warning, emergency instruction, or replacement for government disaster-management authorities.

For real-world deployment, the system would require extensive validation using field observations, regional datasets, expert review, and integration with authorized disaster-management agencies.