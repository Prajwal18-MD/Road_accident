# Execution Guide - Road Accident Safety Intelligence System

This guide provides step-by-step instructions to set up and run the Road Accident Safety Intelligence System with all 7 advanced features.

## Prerequisites

- **Python 3.8 or higher** installed on your system
- **pip** package manager
- **Internet connection** (for initial dependency installation and OpenStreetMap API calls)
- **Windows/Linux/MacOS** operating system

---

## Step 1: Verify Installation Files

Ensure you have all the required files in your project directory (`d:\Road Accident\`):

### Core Files
- ✓ `accident_prediction_india.csv` - Training dataset
- ✓ `train_notebook.py` - Model training script
- ✓ `streamlit_app.py` - Multi-tab web application
- ✓ `requirements.txt` - Python dependencies

### Module Files (newly created)
- ✓ `geospatial_utils.py` - OpenStreetMap integration
- ✓ `safety_scorer.py` - Safety scoring system
- ✓ `explanation_engine.py` - SHAP explanations
- ✓ `route_engine.py` - Route analysis
- ✓ `ambulance_predictor.py` - Emergency response
- ✓ `ai_recommendations.py` - AI recommendations

### Utility Files
- ✓ `model_utils.py` - Feature engineering utilities
- ✓ `location_counts.json` - Location metadata (optional)

---

## Step 2: Install Dependencies

Open a terminal/command prompt in your project directory and run:

```bash
pip install -r requirements.txt
```

**Expected packages** (will be installed):
- pandas, numpy, scikit-learn (ML and data processing)
- streamlit, streamlit-folium (web interface)
- folium (interactive maps)
- xgboost, lightgbm, catboost (advanced ML models)
- shap (model explanations)
- geopy (geocoding)
- requests (API calls)
- matplotlib, seaborn (visualizations)

**Installation time**: 3-5 minutes depending on your internet speed.

**Note**: If you encounter errors with specific packages, you can install them individually:
```bash
pip install streamlit
pip install shap
pip install folium streamlit-folium
pip install requests geopy
```

---

## Step 3: Train the Model (First Time Only)

If you don't have `model.joblib` file, you need to train the model first:

```bash
python train_notebook.py
```

**What this does**:
- Loads the accident dataset (`accident_prediction_india.csv`)
- Performs feature engineering
- Trains a RandomForest classifier
- Saves the trained model as `model.joblib` (~250 MB file)
- Generates `location_counts.json` with accident statistics

**Training time**: 5-10 minutes depending on your CPU.

**Expected output**:
```
Loading data...
Loaded 10000+ accident records
Feature engineering...
Training model...
Model accuracy: 85-90%
Model saved to model.joblib
```

**Note**: You only need to run this once. The `model.joblib` file will be reused.

---

## Step 4: Run the Application

Start the Streamlit web application:

```bash
streamlit run streamlit_app.py
```

**Expected output**:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**What happens**:
- Streamlit server starts on port 8501
- Your default web browser opens automatically
- Application loads all modules and the trained model

**Loading time**: 10-30 seconds (loads large model file)

---

## Step 5: Navigate the Application

The application has **4 main tabs**:

### Tab 1: 🔍 Risk Prediction
**Purpose**: Predict accident risk for a specific location and conditions.

**How to use**:
1. Enter location details (area, coordinates)
2. Set time parameters (hour, day of week)
3. Configure road conditions (type, speed limit, traffic)
4. Select weather conditions
5. Click "🔍 Analyze Risk"

**Output**:
- Risk level (Low/Medium/High) with confidence
- Safety score (0-100)
- SHAP-based explanation showing top contributing factors
- Key metrics visualization

**Example test**:
- Area: "MG Road"
- Time: 8:00 PM, Friday
- Weather: Raining
- Road: Single carriageway, 50 km/h
- Expected: Medium-High risk

---

### Tab 2: 🗺️ Route Analysis
**Purpose**: Compare multiple routes between two locations by safety.

**How to use**:
1. Enter origin address (e.g., "Bangalore Railway Station")
2. Enter destination address (e.g., "Kempegowda Airport")
3. Set travel time and weather
4. Choose number of alternative routes (1-3)
5. Click "🗺️ Analyze Routes"

**Output**:
- Route comparison table (safety score, distance, duration)
- Interactive map with color-coded routes
- Recommendation for safest route
- Detailed analysis for selected route

**Note**: Requires internet connection for OpenStreetMap routing.

**Example test**:
- Origin: "Bangalore City Railway Station"
- Destination: "Bangalore Airport"
- Time: 2:00 PM, Monday
- Expected: 2-3 route alternatives with safety scores

---

### Tab 3: 💡 Safety Recommendations
**Purpose**: Get AI-driven safety improvement suggestions.

**How to use**:
1. Select area/location
2. Enter known accident count
3. Set typical conditions (hour, weather, road type)
4. Click "💡 Generate Recommendations"

**Output**:
- High-priority action items
- Quick wins (fast implementation)
- Recommendations by category (Infrastructure, Enforcement, Signage, etc.)
- Estimated impact and implementation timeline

**Example test**:
- Area: High-accident zone
- Accident count: 15+
- Time: Night (8 PM)
- Weather: Raining
- Expected: 5-10 recommendations including speed breakers, lighting, drainage

---

### Tab 4: 🚑 Emergency Response
**Purpose**: Find nearest hospitals and predict ambulance response times.

**How to use**:
1. Enter accident location coordinates
2. Set time of accident
3. Adjust search radius and max hospitals
4. Click "🚑 Find Nearest Hospitals"

**Output**:
- Recommended hospital with fastest ETA
- Response time breakdown (preparation + travel)
- Survival probability estimate (Golden Hour concept)
- Table of all nearby hospitals ranked by ETA
- Interactive map showing accident location and hospitals

**Note**: Requires internet for hospital search (OpenStreetMap).

**Example test**:
- Location: Central Bangalore (12.9716, 77.5946)
- Time: 8:00 PM
- Expected: 3-5 hospitals with ETAs of 5-20 minutes

---

## Step 6: Testing All Features

### Quick Test Checklist

1. **Basic Risk Prediction** ✓
   - Use default values
   - Verify prediction appears
   - Check if SHAP explanation shows

2. **Route Analysis** ✓
   - Test with known addresses in your area
   - Verify routes appear on map
   - Check safety scores are calculated

3. **Safety Recommendations** ✓
   - Try high-accident area (count > 10)
   - Verify recommendations generate
   - Check priority rankings

4. **Emergency Response** ✓
   - Use coordinates near you
   - Verify hospitals appear
   - Check ETAs are reasonable (5-30 min)

---

## Troubleshooting

### Issue: "Model file not found"
**Solution**: Run `python train_notebook.py` to train the model first.

### Issue: "SHAP not available"
**Solution**: Install SHAP: `pip install shap`. Not critical - basic explanations will work.

### Issue: "No routes found" in Route Analysis
**Solution**: 
- Check internet connection
- Try simpler addresses (e.g., "Bangalore" instead of full street address)
- OSRM public server may be slow - wait and retry

### Issue: "No hospitals found"
**Solution**:
- Increase search radius
- Check internet connection for OpenStreetMap API
- System will show mock hospitals if API fails

### Issue: Application loads slowly
**Solution**: 
- Normal on first load (loads 250MB model)
- Subsequent interactions are faster
- Use smaller dataset for training if needed

### Issue: Geocoding errors
**Solution**:
- Use coordinates instead of addresses
- Nominatim API has rate limits (1 req/sec)
- Results are cached - retry after a few seconds

---

## Performance Notes

### Speed Expectations
- **Model loading**: 10-30 seconds (one-time)
- **Single prediction**: < 1 second
- **Route analysis**: 5-15 seconds (depends on OSM API)
- **Hospital search**: 3-10 seconds (depends on OSM API)
- **Recommendations**: < 1 second

### API Rate Limits
- **Nominatim (geocoding)**: 1 request/second
- **Overpass (hospital search)**: No strict limit, but use reasonably
- **OSRM (routing)**: Public server has fair-use limits

### Optimization Tips
- Results are cached (geocoding, hospitals)
- Restart app to clear cache if needed
- For production use, consider local OSRM installation

---

## Advanced Configuration

### Customizing Safety Scores
Edit `safety_scorer.py` to adjust scoring weights:
```python
# Line 50-60: Adjust point deductions
road_risk = {
    "Highway": -10,  # Change this value
    "Single carriageway": -5,
    ...
}
```

### Changing Risk Thresholds
Edit `model_utils.py`:
```python
def prob_to_risk(probability, low_thresh=0.30, high_thresh=0.60):
    # Adjust thresholds here
```

### Adding Custom Recommendations
Edit `ai_recommendations.py`, function `get_location_recommendations()` to add your own rules.

---

## Data Privacy & Security

- ✓ All ML processing happens **locally** on your machine
- ✓ Only geocoding/routing queries sent to OpenStreetMap (no personal data)
- ✓ No data collection or telemetry
- ✓ Can work offline (except OSM API calls)

---

## Next Steps

1. **Test with real locations** in your area
2. **Customize recommendations** for your region
3. **Integrate with your data** by replacing `accident_prediction_india.csv`
4. **Deploy on server** using `streamlit run --server.port 80`
5. **Share with team** using network URL shown in terminal

---

## Support

For issues or questions:
- Check `technical.md` for algorithm details
- Review `README.md` for feature descriptions
- Inspect console output for error messages
- Ensure all module files are present in the directory

---

## Summary Commands

```bash
# First time setup
pip install -r requirements.txt
python train_notebook.py

# Run application
streamlit run streamlit_app.py

# Access in browser
# Open: http://localhost:8501
```

**Enjoy your Road Safety Intelligence System! 🚦**
