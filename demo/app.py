"""Streamlit demo for sustainable farming recommendation system."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import FarmingDataGenerator
from models import NeuralNetwork


# Page configuration
st.set_page_config(
    page_title="Sustainable Farming Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F0FFF0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .recommendation-card {
        background-color: #E8F5E8;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #2E8B57;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_model_artifacts():
    """Load trained model and artifacts."""
    try:
        # Load scaler
        scaler = joblib.load("assets/scaler.pkl")
        
        # Load feature names and recommendations
        with open("assets/feature_names.json", 'r') as f:
            feature_names = json.load(f)
        
        with open("assets/recommendations.json", 'r') as f:
            recommendations = json.load(f)
        
        # Load model (try different formats)
        model = None
        model_type = None
        
        try:
            # Try loading PyTorch model
            model_state = torch.load("assets/best_model.pth", map_location='cpu')
            model = NeuralNetwork(input_size=4, hidden_sizes=[64, 32], num_classes=4)
            model.load_state_dict(model_state)
            model.eval()
            model_type = 'neural_network'
        except:
            try:
                # Try loading sklearn model
                model = joblib.load("assets/best_model.pkl")
                model_type = 'sklearn'
            except:
                st.error("Could not load trained model. Please run training first.")
                return None, None, None, None, None
        
        return scaler, feature_names, recommendations, model, model_type
        
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None, None, None


def predict_recommendation(features, scaler, model, model_type):
    """Make prediction for given features."""
    try:
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        if model_type == 'neural_network':
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled)
                outputs = model(features_tensor)
                prediction = torch.argmax(outputs, dim=1).item()
        else:
            prediction = model.predict(features_scaled)[0]
        
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🌱 Sustainable Farming Recommendation System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This AI-powered system recommends sustainable farming practices based on soil type, 
    water availability, temperature, and crop type. Get personalized recommendations 
    to optimize your agricultural practices for environmental sustainability.
    """)
    
    # Load model artifacts
    scaler, feature_names, recommendations, model, model_type = load_model_artifacts()
    
    if model is None:
        st.stop()
    
    # Sidebar for input parameters
    st.sidebar.header("🌾 Farm Parameters")
    
    # Soil type selection
    soil_types = ['Sandy', 'Loamy', 'Clay', 'Silt', 'Peat']
    soil_type_idx = st.sidebar.selectbox(
        "Soil Type",
        options=range(len(soil_types)),
        format_func=lambda x: soil_types[x],
        help="Select the predominant soil type in your farm"
    )
    
    # Water availability
    water_availability = st.sidebar.slider(
        "Water Availability (%)",
        min_value=0,
        max_value=100,
        value=60,
        help="Percentage of water availability in your region"
    )
    
    # Average temperature
    avg_temperature = st.sidebar.slider(
        "Average Temperature (°C)",
        min_value=15.0,
        max_value=40.0,
        value=26.0,
        step=0.1,
        help="Average temperature in your farming region"
    )
    
    # Crop type selection
    crop_types = ['Wheat', 'Rice', 'Corn', 'Soybean', 'Vegetables']
    crop_type_idx = st.sidebar.selectbox(
        "Crop Type",
        options=range(len(crop_types)),
        format_func=lambda x: crop_types[x],
        help="Select the crop you plan to grow"
    )
    
    # Create feature vector
    features = np.array([soil_type_idx, water_availability, avg_temperature, crop_type_idx])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Farm Analysis")
        
        # Display input parameters
        st.subheader("Current Farm Parameters")
        
        param_data = {
            'Parameter': ['Soil Type', 'Water Availability', 'Temperature', 'Crop Type'],
            'Value': [
                soil_types[soil_type_idx],
                f"{water_availability}%",
                f"{avg_temperature}°C",
                crop_types[crop_type_idx]
            ]
        }
        
        param_df = pd.DataFrame(param_data)
        st.dataframe(param_df, use_container_width=True)
        
        # Feature visualization
        st.subheader("Parameter Visualization")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Soil Type', 'Water Availability', 'Temperature', 'Crop Type'],
            specs=[[{"type": "bar"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "bar"}]]
        )
        
        # Soil type bar chart
        soil_counts = [1 if i == soil_type_idx else 0 for i in range(len(soil_types))]
        fig.add_trace(
            go.Bar(x=soil_types, y=soil_counts, name="Soil Type", marker_color='#2E8B57'),
            row=1, col=1
        )
        
        # Water availability gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=water_availability,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Water (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "#2E8B57"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 100], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}),
            row=1, col=2
        )
        
        # Temperature gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=avg_temperature,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Temperature (°C)"},
                gauge={'axis': {'range': [15, 40]},
                       'bar': {'color': "#FF6B6B"},
                       'steps': [{'range': [15, 25], 'color': "lightblue"},
                                {'range': [25, 35], 'color': "yellow"},
                                {'range': [35, 40], 'color': "red"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 35}}),
            row=2, col=1
        )
        
        # Crop type bar chart
        crop_counts = [1 if i == crop_type_idx else 0 for i in range(len(crop_types))]
        fig.add_trace(
            go.Bar(x=crop_types, y=crop_counts, name="Crop Type", marker_color='#32CD32'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.header("🎯 Recommendation")
        
        # Get prediction
        if st.button("Get Recommendation", type="primary"):
            prediction = predict_recommendation(features, scaler, model, model_type)
            
            if prediction is not None:
                recommendation_name = recommendations[str(prediction)]
                
                # Display recommendation
                st.markdown(f"""
                <div class="recommendation-card">
                    <h3>🌱 Recommended Practice</h3>
                    <h2 style="color: #2E8B57;">{recommendation_name}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendation details
                st.subheader("Why This Recommendation?")
                
                recommendation_details = {
                    "Crop Rotation": "Improves soil health, reduces pests, and maintains soil fertility through diverse crop cycles.",
                    "Organic Farming": "Reduces chemical inputs, promotes biodiversity, and maintains long-term soil health.",
                    "Drip Irrigation": "Conserves water, reduces evaporation, and provides precise water delivery to plant roots.",
                    "Compost Usage": "Enriches soil with organic matter, improves water retention, and reduces waste."
                }
                
                st.info(recommendation_details.get(recommendation_name, "Sustainable farming practice recommended based on your farm conditions."))
                
                # Additional insights
                st.subheader("💡 Additional Insights")
                
                insights = []
                if water_availability < 50:
                    insights.append("💧 Low water availability detected - consider water conservation practices")
                if avg_temperature > 30:
                    insights.append("🌡️ High temperature detected - consider heat-resistant crop varieties")
                if soil_type_idx == 2:  # Clay
                    insights.append("🏗️ Clay soil detected - good water retention but may need drainage")
                if soil_type_idx == 0:  # Sandy
                    insights.append("🏖️ Sandy soil detected - good drainage but may need water retention")
                
                for insight in insights:
                    st.warning(insight)
        
        # Model information
        st.subheader("🤖 Model Information")
        st.info(f"Model Type: {model_type.replace('_', ' ').title()}")
        st.info("Trained on synthetic farming data with heuristic rules")
        
        # About section
        st.subheader("ℹ️ About")
        st.markdown("""
        This system uses machine learning to recommend sustainable farming practices 
        based on environmental and agricultural parameters. The recommendations are 
        designed to promote environmental sustainability and optimize resource usage.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🌱 Sustainable Farming Recommendation System | Author: <a href="https://github.com/kryptologyst">kryptologyst</a></p>
        <p><em>This is a research demonstration tool. Always consult with agricultural experts for real-world farming decisions.</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
