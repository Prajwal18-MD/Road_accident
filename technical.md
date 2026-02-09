# Technical Documentation

## Problem Definition
Predict the risk level (Low/Medium/High) or severity of road accidents based on input conditions, and provide comprehensive safety intelligence including route analysis, safety recommendations, and emergency response planning.

## Feature Engineering

### Temporal Features
-   **Hour of Day**: Extracted from timestamp. Critical for capturing rush hour vs. late-night patterns.
-   **Is Night**: Boolean flag derived from hour (e.g., 6 PM to 6 AM).
-   **Day of Week**: Weekday vs. Weekend patterns.

### Spatial Features
-   **Clustering**: The notebook includes (optional) KMeans clustering on Latitude/Longitude to create 'Location Cluster' features, which can capture accident hotspots better than raw coordinates for tree-based models.
-   **Historical Accident Count**: Number of past accidents at each location, used as a strong predictor.

### Domain-Specific Mappings
-   **Risk Mapping**: We convert probabilistic outputs into categorical risk levels:
    -   **Low Risk**: Probability < 30%
    -   **Medium Risk**: 30% <= Probability < 60%
    -   **High Risk**: Probability >= 60%
    These thresholds are adjustable in `model_utils.py`.

## Modeling Strategy

### Preprocessing
-   **Numeric**: Simple Imputation (Median) + Scaling (StandardScaler - mostly for linear models, but kept for consistency).
-   **Categorical**: OneHotEncoding for linear models, or direct categorical support for boosted trees (if enabled).

### Recommended Models
1.  **RandomForestClassifier**: Robust baseline, handles non-linearities well, less prone to overfitting than un-tuned boosting. **(Default)**
2.  **XGBoost / LightGBM / CatBoost**: State-of-the-art for tabular data. Code includes commented-out sections to enable these.

### Handling Class Imbalance
-   The training script uses `class_weight='balanced'` in RandomForest to penalize mistakes on minority classes (e.g., fatal accidents).
-   Stratified K-Fold cross-validation ensures representative evaluation/training splits.

## Evaluation Metrics
-   **Accuracy**: General correctness (use with caution on imbalanced data).
-   **Classification Report**: Precision, Recall, F1-Score per class.
-   **Confusion Matrix**: Visual check of misclassifications.
-   **Calibration Curve**: Checks if predicted probabilities match actual frequencies (essential for "Risk" interpretation).

---

## Advanced Features

### 1. Safety Scoring Algorithm

The safety scoring system calculates a 0-100 score representing how safe a location/route is at a given time.

**Algorithm**:
```
Base Score = 100

# Historical Accident Impact
if location has accident history:
    risk_factor = accidents_at_location / max_accidents_globally
    base_score -= risk_factor × 40  # Up to 40 points deduction

# Road Conditions Adjustment
- Highway: -10 points
- Single carriageway: -5 points
- Speed limit >= 100 km/h: -15 points
- Speed limit >= 80 km/h: -10 points
- Speed limit < 40 km/h: +5 points (safer)
- High traffic: -8 points

# Temporal Adjustments
- Night time (6 PM - 6 AM): -12 points
- Very late night (12 AM - 4 AM): -8 additional points
- Rush hour (8-9 AM, 5-7 PM): -5 points
- Weekend: -3 points

# Weather Impact
- Raining: -15 points
- Raining + Windy: -25 points
- Fog/Mist: -20 points

Final Score = clamp(adjusted_score, 0, 100)
```

**Categories**:
- 80-100: Very Safe (Green)
- 60-79: Moderate (Yellow)
- 40-59: Caution (Orange)
- 0-39: High Risk (Red)

### 2. Route Safety Analysis

**Methodology**:
1. Query OSRM API for multiple route alternatives (up to 3 routes)
2. Sample 10-15 evenly-spaced points along each route
3. For each point:
   - Predict accident risk using ML model
   - Calculate safety score considering conditions
4. Aggregate route score: `0.6 × average_score + 0.4 × minimum_score`
   - Weighted toward weakest link (worst segment matters)
5. Rank routes by safety score

**Trade-off Analysis**:
- Compare safety vs. distance
- Identify safest route (may not be shortest)
- Provide balanced recommendation

### 3. SHAP-Based Risk Explanation

Uses SHAP (SHapley Additive exPlanations) for model interpretability.

**Process**:
1. Initialize `TreeExplainer` with trained RandomForest model
2. For each prediction, calculate SHAP values for all features
3. Rank features by absolute SHAP value (impact magnitude)
4. Extract top 5 contributing factors
5. Generate natural language explanation

**Output**:
- Feature importance percentages
- Direction of impact (increases/decreases risk)
- Natural language summary
- Actionable insights based on risk factors

**Fallback**: If SHAP unavailable, use rule-based explanations based on feature values.

### 4. AI-Driven Safety Recommendations

**Pattern Detection**:
- Analyze historical accident data for:
  - Peak accident hours at specific locations
  - Weather-related accident patterns
  - Road type risk correlations
  - Day-of-week trends

**Recommendation Categories**:
1. **Infrastructure**: Speed breakers, lighting, median barriers, road widening
2. **Enforcement**: Traffic police deployment, speed cameras, DUI checkpoints
3. **Signage**: Warning signs, lane markings, reflective markers
4. **Traffic Management**: Signal optimization, lane management, diversion routes
5. **Road Maintenance**: Drainage, anti-skid surfaces, pothole repairs

**Prioritization**:
- High Priority: Immediate safety impact (e.g., high-accident zones)
- Medium Priority: Notable improvement (e.g., lighting upgrades)
- Low Priority: Preventive measures

**Impact Estimation**:
- Very High: 40% accident reduction
- High: 25% reduction
- Medium: 15% reduction
- Low: 5% reduction

### 5. Ambulance Response Time Prediction

**Model**:
```
Total ETA = Preparation Time + Travel Time

Preparation Time = 5-8 minutes (dispatch + ambulance ready)

Travel Time = Distance / Speed

Speed adjustments:
- Night hours: 1.2× faster (less traffic)
- Rush hour: 0.7× slower (congestion)
- Emergency vehicle factor: 0.8× (can bypass some traffic)

Final ETA = Preparation + (Adjusted Travel Time)
```

**Hospital Selection**:
1. Use OpenStreetMap Overpass API to find hospitals within radius
2. Filter for emergency/trauma facilities
3. Calculate route distance using OSRM
4. Estimate ETA considering time-of-day
5. Rank by fastest response time

**Golden Hour Concept**:
- Critical first 60 minutes after accident
- Survival probability decreases significantly after 60 min
- ETA <= 10 min: 95% survival rate
- ETA <= 20 min: 85% survival rate
- ETA > 60 min: 40% survival rate

### 6. Geospatial Data Integration

**APIs Used** (all free, no keys required):
- **Nominatim**: Geocoding (address ↔ coordinates)
- **Overpass API**: Hospital/emergency facility search
- **OSRM**: Route calculation and distance estimation

**Rate Limiting**:
- Nominatim: 1 request/second
- Caching implemented to reduce API calls

**Offline Mode**:
- Haversine distance as fallback for routing
- Mock hospital data if API unavailable
- Average speed estimates without real-time traffic

---

## Data Flow

### Risk Prediction Flow
```
User Input → Feature Engineering → Preprocessing Pipeline → 
Random Forest Model → Risk Probabilities → SHAP Explanation → 
Safety Score Calculation → Display Results
```

### Route Analysis Flow
```
Origin/Destination → Geocoding → OSRM Routing → 
Sample Route Points → Predict Risk for Each Point → 
Aggregate Safety Scores → Rank Routes → Visualize on Map
```

### Emergency Response Flow
```
Accident Location → Find Nearby Hospitals (Overpass API) → 
Calculate Route Distance (OSRM) → Estimate ETA → 
Rank by Response Time → Display with Map
```

---

## Future Improvements
-   **Real-time Traffic**: Integrate with Google Maps/HERE APIs for live traffic data
-   **Deep Learning**: Experiment with neural networks for spatial-temporal patterns
-   **Optuna Tuning**: Use Optuna for Bayesian hyperparameter optimization
-   **SHAP Dashboard**: Interactive SHAP visualization in Streamlit
-   **Geospatial Libraries**: Experiment with H3 indexes for better spatial binning
-   **Mobile App**: React Native wrapper for mobile deployment
-   **Historical Caching**: Cache OSM queries for offline operation
-   **Custom Routing**: Local OSRM installation for full offline capability
