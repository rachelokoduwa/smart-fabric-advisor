# 👔 Smart Fabric Advisor

**AI-Powered Fabric Recommendation System using Machine Learning**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

Smart Fabric Advisor helps users choose the perfect fabric for any occasion based on:
- 📍 **Location** (38 cities worldwide)
- 🌡️ **Weather** (temperature, humidity, precipitation)
- 📅 **Season** (Summer/Winter)
- 🎉 **Event Type** (12 different events)

Powered by a **Gradient Boosting ML model** with **99.81% R² accuracy**.

---

## ✨ Features

- 🤖 **ML-Powered**: Gradient Boosting model trained on 2,496 scenarios
- 🌍 **Global Coverage**: 38 cities across all continents
- 🎨 **Interactive UI**: Built with Streamlit
- 📊 **Visual Analytics**: Charts showing fabric properties
- ⚡ **Real-Time**: Instant predictions (< 1 second)
- 📱 **Responsive**: Works on desktop, tablet, mobile

---

## 🚀 Quick Start

### Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/smart-fabric-advisor.git
cd smart-fabric-advisor
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the app:**
```bash
streamlit run fabric_advisor_app.py
```

4. **Open browser:**
Navigate to `http://localhost:8501`

---

## 📁 Project Structure
```
smart-fabric-advisor/
├── fabric_advisor_app.py       # Main Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── LICENSE                      # MIT License
├── data/                       # Dataset files
│   ├── fabric_properties.csv
│   ├── event_properties.csv
│   ├── climate_zones.csv
│   ├── city_mappings.csv
│   └── main_training_dataset.csv
├── models/                     # Trained ML models
│   ├── gradient_boosting_model.pkl
│   ├── label_encoders.pkl
│   └── feature_info.json
└── images/                     # Visualizations
```

---

## 📊 Model Performance

| Model | Test MAE | Test R² | Training Time |
|-------|----------|---------|---------------|
| Linear Regression | 10.05 | 0.22 | < 1 sec |
| Random Forest | 1.10 | 0.99 | ~30 sec |
| **Gradient Boosting** | **0.46** | **0.998** | ~2 min |

**Champion Model: Gradient Boosting**
- Predictions accurate within 0.46 points on average
- Explains 99.81% of variance in fabric suitability
- Excellent generalization (minimal overfitting)

---

## 💻 Usage

### Using the Web App

1. **Select City**: Choose from 38 worldwide locations
2. **Choose Season**: Summer or Winter
3. **Pick Event**: Wedding, Business Meeting, Hiking, etc.
4. **Get Recommendations**: Click button for instant results
5. **View Results**: See ranked fabrics with scores and properties

### Example Results

**Winter Wedding in Yellowknife, Canada (-40°C):**
- Top Recommendation: **Cashmere** (Score: 89.5)
- Why: Maximum warmth (10/10) + High formality (8/10)

**Summer Beach in Dubai, UAE (40°C):**
- Top Recommendation: **Linen** (Score: 78.9)
- Why: Maximum breathability (10/10) + Low warmth (2/10)

---

## 📚 Dataset

### Coverage
- **Fabrics**: 13 types (Cotton, Wool, Silk, Linen, Polyester, etc.)
- **Events**: 12 types (Wedding, Business Meeting, Hiking, etc.)
- **Cities**: 38 worldwide locations
- **Climate Zones**: 7 zones (Arctic, Tropical, Desert, etc.)
- **Training Data**: 2,496 unique scenarios

### Features (19 total)
- **Categorical (8)**: fabric_name, fabric_category, event_name, climate_zone, season, setting, humidity, precipitation
- **Numerical (11)**: breathability, warmth, formality, water_resistance, event_formality_level, duration_hours, activity_level, weather_exposure, temp_min, temp_max, temp_avg

### Target Variable
- **suitability_score**: 0-100 (fabric's suitability for scenario)

---

## 🛠️ Tech Stack

- **Language**: Python 3.11
- **ML Framework**: scikit-learn (Gradient Boosting)
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Deployment**: Streamlit Cloud

---

## 🔬 Methodology

1. **Data Creation**: Synthetic dataset based on domain knowledge
2. **Feature Engineering**: Temperature calculations, climate mappings
3. **Model Training**: Compared 3 models, selected Gradient Boosting
4. **Evaluation**: MAE, R², overfitting analysis
5. **Deployment**: Streamlit web application

---

## 🔮 Future Improvements

- [ ] Add 100+ cities worldwide
- [ ] Include 20+ fabric types
- [ ] Weather API integration for real-time data
- [ ] User feedback system
- [ ] Outfit recommendations (multiple items)
- [ ] Mobile app version
- [ ] Multi-language support

---

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 👤 Author

**Rachel Okoduwa**
- GitHub: [@rachelokoduwa](https://github.com/rachelokoduwa)
- LinkedIn: [Rachel Okoduwa](https://linkedin.com/in/rachelokoduwa)

---

## 🙏 Acknowledgments

- Climate data based on Köppen climate classification
- Fabric properties from textile industry standards
- Event formality levels from dress code conventions

---

**⭐ If you found this project helpful, please give it a star!**

Built with ❤️ using Python, scikit-learn, and Streamlit
