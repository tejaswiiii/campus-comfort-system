# Smart Campus Comfort & Crowd Prediction

An AI system predicting campus crowd density and environmental comfort for intelligent navigation using Neural Networks and Fuzzy Logic.

## Features

- Deep Neural Network for crowd density prediction
- Fuzzy Logic System for comfort scoring (13 rules)
- Real-time weather integration via WeatherAPI
- Campus zone mapping via OpenStreetMap/OSMnx
- Smart walking recommendations based on live conditions

## Dataset

Stanford University:
```
Zones:       542 campus buildings/areas
Time range:  72 hours (3 days)
Total rows:  39,024
```

Key features: zone type, shade score, temperature, humidity, wind speed, hour of day.

## Program Requirements
```bash
pip install tensorflow scikit-learn scikit-fuzzy pandas numpy osmnx geopandas plotly requests streamlit
```

## Demo

A Streamlit dashboard for real-time comfort recommendations:
```bash
streamlit run streamlit_app.py
```

Select a campus zone and time to get comfort scores, crowd predictions, and optimal walking windows.

## How it works

1. Extract campus building data via OSMnx
2. Fetch real-time weather from WeatherAPI
3. Neural Network predicts crowd density per zone (embedding + dense layers)
4. Heat stress calculated using NOAA Heat Index formula
5. Fuzzy Logic combines crowd, heat stress, and shade into a comfort score (0–100)
6. Recommendation engine surfaces best zones and times

## Project structure
```
notebooks/          Main Colab notebook
src/                Data collection, model, fuzzy logic, recommendations
models/             Trained neural network and zone encoder
data/               Zone, weather, and feature datasets
results/            Visualizations and evaluation outputs
streamlit_app.py    Dashboard
```

## Model performance
```
Crowd prediction validation loss:  ~0.02
Comfort score range:               55 – 100
Zone differentiation:              45-point comfort spread
```

## Tech stack

- Python 3.8+
- TensorFlow 2.x
- scikit-fuzzy
- OSMnx / GeoPandas
- Plotly, Streamlit

## License

MIT
