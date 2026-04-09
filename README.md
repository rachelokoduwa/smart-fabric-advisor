# Smart Fabric Advisor

**AI-Powered Fabric Recommendation System using Machine Learning**

---

## Overview

This project started with a simple question: can a machine learning model tell you what fabric to wear based on where you are, what season it is, and what you are doing?

Built over 7 days in January 2026, Smart Fabric Advisor recommends the most suitable fabric for any occasion based on location, weather, season, and event type. It is powered by a Gradient Boosting ML model with 99.81% R² accuracy.

---

## Live Demo

https://smart-fabric-advisor-nkvgpduotpdvegvyfqop8a.streamlit.app/

---

## How It Works

The recommendation score (0-100) is a weighted combination of four factors:

- **Temperature comfort (50%)** — how well the fabric handles the climate
- **Activity suitability (20%)** — how practical the fabric is for the event
- **Formality match (20%)** — how well the fabric matches the event dress code
- **Weather protection (10%)** — how well the fabric handles humidity and rain

---

## Features

- 38 cities worldwide
- 12 event types
- 13 fabric types
- Real-time ML predictions in under 1 second
- Interactive charts and property breakdowns

---

## Model Performance

| Model | Test MAE | Test R² | Training Time |
|-------|----------|---------|---------------|
| Linear Regression | 10.05 | 0.22 | < 1 sec |
| Random Forest | 1.10 | 0.99 | ~30 sec |
| **Gradient Boosting** | **0.46** | **0.998** | ~2 min |

The Gradient Boosting model was selected as the final model. It predicts fabric suitability within 0.46 points on average and explains 99.81% of variance in the data with minimal overfitting.

---

## Dataset

The training dataset was built from scratch using domain knowledge across five dimensions:

- **Fabrics**: 13 types with properties including breathability, warmth, formality, and water resistance
- **Events**: 12 types with formality levels, activity requirements, and weather exposure
- **Cities**: 38 worldwide locations mapped to 7 climate zones
- **Seasons**: Summer and Winter conditions for each climate zone
- **Training scenarios**: 2,496 unique combinations

---

## Quick Start

```bash
git clone https://github.com/rachelokoduwa/smart-fabric-advisor.git
cd smart-fabric-advisor
pip install -r requirements.txt
streamlit run fabric_advisor_app.py
```

---

## Tech Stack

- Python 3.11
- scikit-learn (Gradient Boosting, Label Encoding)
- Streamlit
- pandas, numpy
- matplotlib, seaborn
- joblib

---

## Project Structure
smart-fabric-advisor/
├──fabric_advisor_app.py 

# Main Streamlit application
├── requirements.txt
├── data/
│   ├── fabric_properties.csv
│   ├── event_properties.csv
│   ├── climate_zones.csv
│   ├── city_mappings.csv
│   └── main_training_dataset.csv
├── model/
│   ├── gradient_boosting_model_JOBLIB.pkl
│   ├── label_encoders_JOBLIB.pkl
│   └── feature_info_JOBLIB.json
└── images/

---

## Author

**Rachel Okoduwa**
- GitHub: [rachelokoduwa](https://github.com/rachelokoduwa)
- LinkedIn: [Rachel Okoduwa](https://linkedin.com/in/rachel-okoduwa)
- Portfolio: [rachelokoduwa.github.io](https://rachelokoduwa.github.io)

---

*Built with curiosity and a lot of help from Claude AI.*
