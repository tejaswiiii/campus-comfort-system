import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tensorflow as tf
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Smart Campus Comfort System",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .comfort-high {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        color: black;
    }
    .comfort-medium {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        color: black;
    }
    .comfort-low {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all CSV files"""
    try:
        df = pd.read_csv("features_with_comfort_final.csv", parse_dates=['timestamp','timestamp_hour'])
        zones = pd.read_csv("zones.csv")
        return df, zones
    except FileNotFoundError as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("📋 Please ensure these files are in the same directory as the app:")
        st.code("- features_with_comfort_final.csv\n- zones.csv")
        st.stop()

@st.cache_resource
def load_model():
    """Load trained neural network model"""
    try:
        model = tf.keras.models.load_model("crowd_model.keras")
        label_encoder = joblib.load("label_encoder_zone.joblib")
        return model, label_encoder
    except Exception as e:
        st.warning(f"⚠️ Model not loaded: {e}")
        return None, None

# Load data
df, zones = load_data()
model, label_encoder = load_model()

def calculate_walking_time(origin_zone, dest_zone, zones_df):
    """
    Calculate realistic walking time between two zones on campus.
    Uses Haversine formula for distance, assumes 5 km/h walking speed.
    """
    try:
        origin_data = zones_df[zones_df['zone_name'] == origin_zone].iloc[0]
        dest_data = zones_df[zones_df['zone_name'] == dest_zone].iloc[0]
       
        # Haversine formula
        lat1, lon1 = origin_data['lat'], origin_data['lon']
        lat2, lon2 = dest_data['lat'], dest_data['lon']
       
        R = 6371
       
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
       
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
       
        distance_km = R * c
        distance_m = distance_km * 1000
        walking_speed = 83.3
        walk_time_minutes = distance_m / walking_speed
        walk_time_minutes += 1.5
       
        return max(1, int(walk_time_minutes))  
    except:
        return 5  # Default fallback

def get_comfort_color(score):
    """Return color based on comfort score"""
    if score >= 75:
        return "#28a745"  
    elif score >= 50:
        return "#ffc107"  
    elif score >= 30:
        return "#fd7e14"  
    else:
        return "#dc3545"  

def get_comfort_label(score):
    """Return label based on comfort score"""
    if score >= 80:
        return "Very High"
    elif score >= 60:
        return "High"
    elif score >= 40:
        return "Medium"
    elif score >= 20:
        return "Low"
    else:
        return "Very Low"

def best_zone_now(df_current):
    """Find the zone with highest comfort score at current time"""
    if df_current.empty:
        return None
   
    best_idx = df_current['comfort_score'].idxmax()
    best_row = df_current.loc[best_idx]
   
    return {
        'zone_name': best_row['zone_name'],
        'comfort_score': float(best_row['comfort_score']),
        'predicted_crowd': float(best_row['predicted_crowd']),
        'heat_stress': float(best_row['heat_stress']),
        'shade_score': float(best_row['shade_score'])
    }

def worst_zones_now(df_current, n=5):
    """Find the N worst zones to avoid right now"""
    if df_current.empty:
        return []
   
    worst = df_current.nsmallest(n, 'comfort_score')
    return worst[['zone_name', 'comfort_score', 'predicted_crowd', 'heat_stress']].to_dict('records')

def best_time_to_walk(df_all, origin_zone, dest_zone, start_time, zones_df):
    """
    Find time to walk based on route comfort.
    Returns single best option.
    """
    start = pd.to_datetime(start_time)
   
    walk_minutes = calculate_walking_time(origin_zone, dest_zone, zones_df)
    times = sorted(df_all['timestamp_hour'].unique())
    candidates = [t for t in times if start <= t <= start + timedelta(hours=3)]
   
    if not candidates:
        return None
   
    best_result = None
    best_score = -1
   
    for departure_time in candidates:
        actual_arrival_time = departure_time + timedelta(minutes=walk_minutes)
        departure_data = df_all[df_all['timestamp_hour'] == departure_time]
        available_times = df_all['timestamp_hour'].unique()
        closest_arrival = min(available_times, key=lambda x: abs((x - actual_arrival_time).total_seconds()))
        arrival_data = df_all[df_all['timestamp_hour'] == closest_arrival]
        origin_departure = departure_data[departure_data['zone_name'] == origin_zone]
        dest_arrival = arrival_data[arrival_data['zone_name'] == dest_zone]
       
        if origin_departure.empty or dest_arrival.empty:
            continue
       
        # Calculate route comfort (weighted: 40% departure, 60% arrival)
        origin_comfort = float(origin_departure['comfort_score'].iloc[0])
        dest_comfort = float(dest_arrival['comfort_score'].iloc[0])
       
        route_comfort = 0.4 * origin_comfort + 0.6 * dest_comfort
       
        if route_comfort > best_score:
            best_score = route_comfort
            minutes_from_now = int((departure_time - start).total_seconds() / 60)
            
            best_result = {
                'departure_time': departure_time,
                'arrival_time': closest_arrival,
                'comfort_score': route_comfort,
                'minutes_from_now': minutes_from_now,
                'walk_duration': walk_minutes,
                'origin_comfort': origin_comfort,
                'dest_comfort': dest_comfort
            }
   
    return best_result

def zone_forecast(df_all, zone_name, start_time, hours=6):
    """Get comfort forecast for a specific zone over next N hours"""
    start = pd.to_datetime(start_time)
    end = start + timedelta(hours=hours)
   
    forecast = df_all[
        (df_all['zone_name'] == zone_name) &
        (df_all['timestamp_hour'] >= start) &
        (df_all['timestamp_hour'] <= end)
    ].sort_values('timestamp_hour')
   
    return forecast

# sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Smart+Campus", use_container_width=True)
   
    st.markdown("### ⚙️ Settings")
   
    # Time selector
    available_times = sorted(df['timestamp_hour'].unique())
   
    selected_time = st.selectbox(
        "🕐 Select Time",
        available_times,
        format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M")
    )
   
    # Zone selector
    all_zones = sorted(df['zone_name'].unique())
   
    st.markdown("---")
    st.markdown("### 📊 System Info")
    st.metric("Total Zones", len(all_zones))
    st.metric("Data Points", len(df))
    st.metric("Time Range", f"{len(available_times)} hours")
   
    st.markdown("---")
    st.markdown("### 🎯 About")
    st.info("""
    **Smart Campus Comfort System**
   
    Combines Neural Networks and Fuzzy Logic to predict:
    - Crowd density
    - Heat stress
    - Comfort scores
    - Best times to walk
    """)

# main
st.markdown('<div class="main-header">🏫 Smart Campus Comfort & Crowd Prediction System</div>', unsafe_allow_html=True)

# Get current time data
df_current = df[df['timestamp_hour'] == selected_time].copy()

if df_current.empty:
    st.error("No data available for selected time")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Dashboard",
    "🗺️ Zone Explorer",
    "🚶 Walk Planner",
    "📈 Analytics"
])

#Dashboard

with tab1:
    st.markdown(f"### 📅 Current Time: {pd.to_datetime(selected_time).strftime('%A, %B %d, %Y - %H:%M')}")
   
    # Best zone recommendation
    best_zone = best_zone_now(df_current)
   
    if best_zone:
        col1, col2 = st.columns([2, 1])
       
        with col1:
            comfort_class = "comfort-high" if best_zone['comfort_score'] >= 70 else "comfort-medium" if best_zone['comfort_score'] >= 40 else "comfort-low"
           
            st.markdown(f"""
            <div class="{comfort_class}">
                <h2>🏆 Best Zone Right Now</h2>
                <h3>{best_zone['zone_name']}</h3>
                <h1>{best_zone['comfort_score']:.1f}/100</h1>
                <p><strong>Comfort Level:</strong> {get_comfort_label(best_zone['comfort_score'])}</p>
            </div>
            """, unsafe_allow_html=True)
       
        with col2:
            st.markdown("### 📊 Details")
            st.metric("Crowd Level", f"{best_zone['predicted_crowd']*100:.0f}%")
            st.metric("Heat Stress", f"{best_zone['heat_stress']:.0f}/100")
            st.metric("Shade Coverage", f"{best_zone['shade_score']*100:.0f}%")
   
    st.markdown("---")
   
    # Top comfort zones
    col1, col2 = st.columns(2)
   
    with col1:
        st.markdown("### ✅ Top 5 Most Comfortable Zones")
        top_zones = df_current.nlargest(5, 'comfort_score')[['zone_name', 'comfort_score', 'predicted_crowd']]
       
        for idx, row in top_zones.iterrows():
            crowd_pct = row['predicted_crowd'] * 100
            color = get_comfort_color(row['comfort_score'])
           
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 0.5rem; margin: 0.5rem 0; border-radius: 5px; border-left: 4px solid {color}">
                <strong>{row['zone_name']}</strong><br>
                Comfort: {row['comfort_score']:.1f} | Crowd: {crowd_pct:.0f}%
            </div>
            """, unsafe_allow_html=True)
   
    with col2:
        st.markdown("### ⚠️ Zones to Avoid")
        avoid_zones = worst_zones_now(df_current, n=5)
       
        for zone in avoid_zones:
            crowd_pct = zone['predicted_crowd'] * 100
            st.markdown(f"""
            <div style="background-color: #dc354520; padding: 0.5rem; margin: 0.5rem 0; border-radius: 5px; border-left: 4px solid #dc3545">
                <strong>{zone['zone_name']}</strong><br>
                Comfort: {zone['comfort_score']:.1f} | Crowd: {crowd_pct:.0f}%
            </div>
            """, unsafe_allow_html=True)
   
    st.markdown("---")
   
    # Comfort distribution chart
    st.markdown("### 📊 Campus Comfort Distribution")
   
    fig = px.bar(
        df_current.nlargest(20, 'comfort_score'),
        x='zone_name',
        y='comfort_score',
        color='comfort_score',
        color_continuous_scale=['red', 'yellow', 'green'],
        range_color=[0, 100],
        labels={'zone_name': 'Zone', 'comfort_score': 'Comfort Score'}
    )
   
    fig.update_layout(
        height=400,
        xaxis_tickangle=-45,
        showlegend=False
    )
   
    st.plotly_chart(fig, use_container_width=True)

#Zone explorer

with tab2:
    st.markdown("### 🗺️ Explore Individual Zones")
   
    selected_zone = st.selectbox("Select a zone to analyze:", all_zones)
   
    if selected_zone:
        zone_data_current = df_current[df_current['zone_name'] == selected_zone].iloc[0]
       
        # Current status
        col1, col2, col3, col4 = st.columns(4)
       
        comfort_score = zone_data_current['comfort_score']
        comfort_color = get_comfort_color(comfort_score)
       
        with col1:
            st.metric(
                "Comfort Score",
                f"{comfort_score:.1f}",
                delta=get_comfort_label(comfort_score)
            )
       
        with col2:
            crowd_pct = zone_data_current['predicted_crowd'] * 100
            st.metric("Crowd Level", f"{crowd_pct:.0f}%")
       
        with col3:
            st.metric("Heat Stress", f"{zone_data_current['heat_stress']:.0f}/100")
       
        with col4:
            shade_pct = zone_data_current['shade_score'] * 100
            st.metric("Shade Coverage", f"{shade_pct:.0f}%")
       
        st.markdown("---")
       
        # 6 hour forecast
        st.markdown("### 📈 6-Hour Forecast")
       
        forecast_data = zone_forecast(df, selected_zone, selected_time, hours=6)
       
        if not forecast_data.empty:
            fig = go.Figure()
           
            fig.add_trace(go.Scatter(
                x=forecast_data['timestamp_hour'],
                y=forecast_data['comfort_score'],
                name='Comfort Score',
                line=dict(color='green', width=3),
                fill='tozeroy'
            ))
           
            fig.add_trace(go.Scatter(
                x=forecast_data['timestamp_hour'],
                y=forecast_data['predicted_crowd'] * 100,
                name='Crowd Level (%)',
                line=dict(color='blue', width=2)
            ))
           
            fig.add_trace(go.Scatter(
                x=forecast_data['timestamp_hour'],
                y=forecast_data['heat_stress'],
                name='Heat Stress',
                line=dict(color='red', width=2)
            ))
           
            fig.update_layout(
                height=400,
                hovermode='x unified',
                yaxis=dict(range=[0, 100])
            )
           
            st.plotly_chart(fig, use_container_width=True)
           
            # Data table
            with st.expander("📋 View Detailed Forecast Data"):
                forecast_display = forecast_data[['timestamp_hour', 'predicted_crowd', 'heat_stress', 'comfort_score']].copy()
                forecast_display['predicted_crowd'] = (forecast_display['predicted_crowd'] * 100).round(0).astype(int)
                forecast_display.columns = ['Time', 'Crowd %', 'Heat Stress', 'Comfort']
                st.dataframe(forecast_display, use_container_width=True)
        else:
            st.warning("No forecast data available for this zone")

#Walk Planner

with tab3:
    st.markdown("### 🚶 Plan Your Walk")
    st.info("Find the time to walk between two locations based on distance and comfort")
   
    col1, col2 = st.columns(2)
   
    with col1:
        origin = st.selectbox("From:", all_zones, key='origin')
   
    with col2:
        destination = st.selectbox("To:", [z for z in all_zones if z != origin], key='dest')
   
    if st.button("🔍 Get Walking Recommendation", type="primary"):
        result = best_time_to_walk(df, origin, destination, selected_time, zones)
       
        if result:
            departure_dt = pd.to_datetime(result['departure_time'])
            actual_arrival_dt = departure_dt + timedelta(minutes=result['walk_duration'])
            
            comfort_class = "comfort-high" if result['comfort_score'] >= 70 else "comfort-medium" if result['comfort_score'] >= 40 else "comfort-low"
            
            # Show actual clock time instead of "in X hours"
            st.markdown(f"""
            <div class="{comfort_class}">
                <h3>🎯 Time Taken to Walk</h3>
                <h2>Depart at {departure_dt.strftime('%I:%M %p')}</h2>
                <p style="font-size: 1.1rem;">
                    Arrive at {actual_arrival_dt.strftime('%I:%M %p')} ({result['walk_duration']} min walk)
                </p>
                <p><strong>Expected Route Comfort:</strong> {result['comfort_score']:.1f}/100 - {get_comfort_label(result['comfort_score'])}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 📊 Route Conditions")
            
            departure_data = df[df['timestamp_hour'] == result['departure_time']]
            arrival_data = df[df['timestamp_hour'] == result['arrival_time']]
            
            origin_data = departure_data[departure_data['zone_name'] == origin].iloc[0]
            dest_data = arrival_data[arrival_data['zone_name'] == destination].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**{origin}** (at departure)")
                st.metric("Comfort", f"{origin_data['comfort_score']:.1f}")
                st.metric("Crowd", f"{origin_data['predicted_crowd']*100:.0f}%")
                st.metric("Heat Stress", f"{origin_data['heat_stress']:.0f}/100")
            
            with col2:
                st.markdown(f"**{destination}** (at arrival)")
                st.metric("Comfort", f"{dest_data['comfort_score']:.1f}")
                st.metric("Crowd", f"{dest_data['predicted_crowd']*100:.0f}%")
                st.metric("Heat Stress", f"{dest_data['heat_stress']:.0f}/100")
       
        else:
            st.error("❌ No suitable walking time found in the next 3 hours.")

#Analytics

with tab4:
    st.markdown("### 📊 System Analytics & Insights")
   
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
   
    with col1:
        avg_comfort = df_current['comfort_score'].mean()
        st.metric("Average Comfort", f"{avg_comfort:.1f}")
   
    with col2:
        avg_crowd = df_current['predicted_crowd'].mean() * 100
        st.metric("Average Crowd", f"{avg_crowd:.0f}%")
   
    with col3:
        avg_heat = df_current['heat_stress'].mean()
        st.metric("Average Heat Stress", f"{avg_heat:.0f}")
   
    with col4:
        comfortable_zones = (df_current['comfort_score'] >= 70).sum()
        st.metric("Comfortable Zones", f"{comfortable_zones}/{len(df_current)}")
   
    st.markdown("---")
   
    # Comfort distribution
    col1, col2 = st.columns(2)
   
    with col1:
        st.markdown("#### Comfort Score Distribution")
       
        bins = [0, 20, 40, 60, 80, 100]
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        df_current['comfort_category'] = pd.cut(df_current['comfort_score'], bins=bins, labels=labels)
       
        dist_data = df_current['comfort_category'].value_counts().sort_index()
       
        fig = px.pie(
            values=dist_data.values,
            names=dist_data.index,
            color_discrete_sequence=px.colors.diverging.RdYlGn
        )
       
        st.plotly_chart(fig, use_container_width=True)
   
    with col2:
        st.markdown("#### Crowd vs Comfort Correlation")
       
        fig = px.scatter(
            df_current,
            x='predicted_crowd',
            y='comfort_score',
            color='heat_stress',
            size='shade_score',
            hover_data=['zone_name'],
            color_continuous_scale='Reds',
            labels={
                'predicted_crowd': 'Crowd Level',
                'comfort_score': 'Comfort Score',
                'heat_stress': 'Heat Stress'
            }
        )
       
        st.plotly_chart(fig, use_container_width=True)
   
    st.markdown("---")
   
    # Time-based analysis
    st.markdown("### 📈 Time-Based Analysis")
   
    hourly_stats = df.groupby('hour_of_day').agg({
        'comfort_score': 'mean',
        'predicted_crowd': 'mean',
        'heat_stress': 'mean'
    }).reset_index()
   
    fig = go.Figure()
   
    fig.add_trace(go.Scatter(
        x=hourly_stats['hour_of_day'],
        y=hourly_stats['comfort_score'],
        name='Avg Comfort',
        line=dict(color='green', width=3)
    ))
   
    fig.add_trace(go.Scatter(
        x=hourly_stats['hour_of_day'],
        y=hourly_stats['predicted_crowd'] * 100,
        name='Avg Crowd %',
        line=dict(color='blue', width=2)
    ))
   
    fig.add_trace(go.Scatter(
        x=hourly_stats['hour_of_day'],
        y=hourly_stats['heat_stress'],
        name='Avg Heat Stress',
        line=dict(color='red', width=2)
    ))
   
    fig.update_layout(
        height=400,
        xaxis_title="Hour of Day",
        yaxis_title="Score",
        hovermode='x unified'
    )
   
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>Smart Campus Comfort System</p>
</div>
""", unsafe_allow_html=True)