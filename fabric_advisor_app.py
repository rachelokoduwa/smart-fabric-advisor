
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Smart Fabric Advisor",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_resource
def load_models_and_data():
    """Load all necessary files"""
    fabric_df = pd.read_csv('fabric_properties.csv')
    event_df = pd.read_csv('event_properties.csv')
    climate_df = pd.read_csv('climate_zones.csv')
    city_df = pd.read_csv('city_mappings.csv')

    with open('gradient_boosting_model.pkl', 'rb') as f:
        ml_model = pickle.load(f)

    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    with open('feature_info.json', 'r') as f:
        feature_info = json.load(f)

    return fabric_df, event_df, climate_df, city_df, ml_model, label_encoders, feature_info

# Load everything
fabric_df, event_df, climate_df, city_df, ml_model, label_encoders, feature_info = load_models_and_data()

# Prediction function
def get_predictions(city, season, event):
    """Get ML predictions for all fabrics"""

    climate_zone = city_df[city_df['city'] == city]['climate_zone'].values[0]
    climate_info = climate_df[climate_df['zone_name'] == climate_zone].iloc[0]
    event_info = event_df[event_df['event_name'] == event].iloc[0]

    if season == 'Winter':
        temp_min = climate_info['winter_temp_min']
        temp_max = climate_info['winter_temp_max']
        humidity = climate_info['winter_humidity']
        precipitation = climate_info['winter_precipitation']
    else:
        temp_min = climate_info['summer_temp_min']
        temp_max = climate_info['summer_temp_max']
        humidity = climate_info['summer_humidity']
        precipitation = climate_info['summer_precipitation']

    temp_avg = (temp_min + temp_max) / 2
    setting = event_info['setting']

    predictions = []

    for idx, fabric_row in fabric_df.iterrows():
        scenario = {
            'fabric_name': fabric_row['fabric_name'],
            'fabric_category': fabric_row['category'],
            'event_name': event,
            'climate_zone': climate_zone,
            'season': season.lower(),
            'setting': setting,
            'humidity': humidity,
            'precipitation': precipitation,
            'breathability': fabric_row['breathability'],
            'warmth': fabric_row['warmth'],
            'formality': fabric_row['formality'],
            'water_resistance': fabric_row['water_resistance'],
            'event_formality_level': event_info['formality_level'],
            'duration_hours': event_info['duration_hours'],
            'activity_level': event_info['activity_level'],
            'weather_exposure': event_info['weather_exposure'],
            'temp_min': temp_min,
            'temp_max': temp_max,
            'temp_avg': temp_avg
        }

        encoded_scenario = scenario.copy()
        for cat_feature in feature_info['categorical_features']:
            encoder = label_encoders[cat_feature]
            value = scenario[cat_feature]
            encoded_scenario[f'{cat_feature}_encoded'] = encoder.transform([value])[0]

        feature_values = [encoded_scenario[feature] for feature in feature_info['feature_columns']]
        X_new = np.array(feature_values).reshape(1, -1)
        predicted_score = ml_model.predict(X_new)[0]

        if predicted_score >= 78:
            category = "Excellent"
        elif predicted_score >= 65:
            category = "Good"
        elif predicted_score >= 50:
            category = "Fair"
        else:
            category = "Poor"

        predictions.append({
            'fabric_name': fabric_row['fabric_name'],
            'fabric_category': fabric_row['category'],
            'predicted_score': predicted_score,
            'category': category,
            'breathability': fabric_row['breathability'],
            'warmth': fabric_row['warmth'],
            'formality': fabric_row['formality'],
            'water_resistance': fabric_row['water_resistance']
        })

    results_df = pd.DataFrame(predictions).sort_values('predicted_score', ascending=False)

    return results_df, {
        'city': city,
        'climate_zone': climate_zone,
        'season': season,
        'event': event,
        'temp_range': f"{temp_min}°C to {temp_max}°C",
        'humidity': humidity,
        'precipitation': precipitation
    }

# App layout
st.title("👔 Smart Fabric Advisor")
st.markdown("### AI-Powered Fabric Recommendations for Any Occasion")
st.markdown("*Powered by Machine Learning (99.81% Accuracy)*")

st.markdown("---")

# Sidebar inputs
st.sidebar.header("📋 Your Event Details")

city = st.sidebar.selectbox(
    "📍 Select City",
    options=sorted(city_df['city'].tolist()),
    index=sorted(city_df['city'].tolist()).index('Toronto, Canada')
)

season = st.sidebar.selectbox(
    "📅 Select Season",
    options=['Summer', 'Winter']
)

event = st.sidebar.selectbox(
    "🎉 Select Event",
    options=sorted(event_df['event_name'].tolist()),
    index=sorted(event_df['event_name'].tolist()).index('Business Meeting')
)

analyze_button = st.sidebar.button("🎯 Get Recommendations", type="primary")

# Main content
if analyze_button:
    with st.spinner('🔄 Analyzing best fabrics for your event...'):
        results_df, metadata = get_predictions(city, season, event)

    # Display scenario info
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📍 Location", metadata['city'])
    with col2:
        st.metric("🌡️ Temperature", metadata['temp_range'])
    with col3:
        st.metric("📅 Season", metadata['season'])
    with col4:
        st.metric("🎉 Event", metadata['event'])

    st.markdown("---")

    # Top recommendations
    st.subheader("🏆 Top Fabric Recommendations")

    top_5 = results_df.head(5)

    for idx, row in top_5.iterrows():
        emoji = "✅" if row['category'] == "Excellent" else "👍" if row['category'] == "Good" else "⚠️"

        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown(f"### {emoji} {row['fabric_name']} ({row['fabric_category']})")
            with col2:
                st.metric("Score", f"{row['predicted_score']:.1f}")
            with col3:
                st.metric("Rating", row['category'])

            # Property bars
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.progress(row['breathability']/10, text=f"Breathability: {row['breathability']}/10")
            with col_b:
                st.progress(row['warmth']/10, text=f"Warmth: {row['warmth']}/10")
            with col_c:
                st.progress(row['formality']/10, text=f"Formality: {row['formality']}/10")
            with col_d:
                st.progress(row['water_resistance']/10, text=f"Water Resist: {row['water_resistance']}/10")

            st.markdown("---")

    # Visualization
    st.subheader("📊 Detailed Analysis")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Score chart
    top_8 = results_df.head(8)
    colors = ['#2ecc71' if score >= 78 else '#3498db' if score >= 65 else '#f39c12'
              for score in top_8['predicted_score']]

    ax1.barh(top_8['fabric_name'], top_8['predicted_score'], color=colors, edgecolor='black')
    ax1.set_xlabel('Recommendation Score', fontweight='bold')
    ax1.set_title('Top 8 Fabric Scores', fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.grid(axis='x', alpha=0.3)

    # Category distribution
    category_counts = results_df['category'].value_counts()
    colors_pie = {'Excellent': '#2ecc71', 'Good': '#3498db', 'Fair': '#f39c12', 'Poor': '#e74c3c'}
    pie_colors = [colors_pie.get(cat, '#95a5a6') for cat in category_counts.index]

    ax2.pie(category_counts.values, labels=category_counts.index, autopct='%1.0f%%',
            colors=pie_colors, textprops={'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black'})
    ax2.set_title('Category Distribution', fontweight='bold')

    st.pyplot(fig)

    # Full results table
    with st.expander("📋 View All Fabric Rankings"):
        st.dataframe(
            results_df[['fabric_name', 'predicted_score', 'category', 'breathability', 'warmth', 'formality', 'water_resistance']],
            use_container_width=True
        )

else:
    st.info("👈 Select your event details in the sidebar and click 'Get Recommendations' to start!")

    st.markdown("### How It Works")
    st.markdown("""
    1. **Select your location** - Choose from 38 cities worldwide
    2. **Pick the season** - Winter or Summer
    3. **Choose your event** - From casual hangouts to formal weddings
    4. **Get instant recommendations** - AI analyzes temperature, formality, and activity level

    Our ML model considers:
    - 🌡️ Temperature and climate
    - 👔 Event formality requirements
    - 💧 Weather conditions (humidity, precipitation)
    - 🏃 Activity level
    - 🧵 Fabric properties (breathability, warmth, water resistance)
    """)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit • Powered by Gradient Boosting ML Model (99.81% R² accuracy)*")
