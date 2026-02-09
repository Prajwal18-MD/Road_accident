# Project Structure

```text
d:\Road Accident\
│
├── accident_prediction_india.csv  # Input Dataset (User provided)
├── model.joblib                   # Trained model pipeline (Created after running the notebook)
│
├── train_notebook.py              # Training script (Jupytext format)
├── streamlit_app.py               # Enhanced multi-tab inference application
├── model_utils.py                 # Shared utility functions and classes
│
├── geospatial_utils.py            # OpenStreetMap integration (geocoding, hospitals, routing)
├── safety_scorer.py               # Safety scoring system (0-100 scale)
├── explanation_engine.py          # SHAP-based risk explanations
├── route_engine.py                # Multi-route analysis and comparison
├── ambulance_predictor.py         # Emergency response time predictions
├── ai_recommendations.py          # AI-driven safety improvement suggestions
│
├── requirements.txt               # Dependencies
├── README.md                      # Project overview
├── guide.md                       # User manual and setup guide
├── technical.md                   # Technical details
├── project_structure.md           # This file
├── .gitignore                     # Git ignore rules
└── example_input.json             # Example JSON input
```
