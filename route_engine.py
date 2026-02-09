"""
Route Analysis Engine

Analyzes and compares multiple routes for safety:
- Multi-route generation using OpenStreetMap
- Route safety scoring based on accident risk
- Route comparison and recommendations
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from geospatial_utils import get_alternative_routes, sample_points_along_route, haversine_distance
from safety_scorer import SafetyScorer


class RouteAnalyzer:
    """Analyze and compare routes for safety."""
    
    def __init__(self, model_pipeline, safety_scorer: SafetyScorer):
        """
        Initialize route analyzer.
        
        Args:
            model_pipeline: Trained accident prediction model
            safety_scorer: SafetyScorer instance for scoring calculations
        """
        self.pipeline = model_pipeline
        self.scorer = safety_scorer
    
    def analyze_route(self, start_lat: float, start_lon: float,
                     end_lat: float, end_lon: float,
                     hour: int = 12, day_of_week: str = "Monday",
                     weather: str = "Fine", num_alternatives: int = 2) -> List[Dict]:
        """
        Analyze multiple routes between two points and rank by safety.
        
        Args:
            start_lat, start_lon: Origin coordinates
            end_lat, end_lon: Destination coordinates
            hour: Hour of day (0-23)
            day_of_week: Day name
            weather: Weather conditions
            num_alternatives: Number of alternative routes to find
            
        Returns:
            List of route dictionaries sorted by safety score (safest first)
        """
        # Get alternative routes from OSM
        routes = get_alternative_routes(start_lat, start_lon, end_lat, end_lon, num_alternatives)
        
        if not routes:
            return []
        
        analyzed_routes = []
        
        for route in routes:
            # Sample points along the route
            geometry = route.get('geometry', [])
            sampled_points = sample_points_along_route(geometry, num_points=15)
            
            # Calculate risk for each point
            risk_scores = []
            safety_scores = []
            
            for lat, lon in sampled_points:
                # Predict accident risk at this point
                risk_pred = self._predict_point_risk(
                    lat, lon, hour, day_of_week, weather
                )
                risk_scores.append(risk_pred['risk_probability'])
                
                # Calculate safety score
                safety_result = self.scorer.calculate_safety_score(
                    area="Route Segment",
                    road_type="Single carriageway",  # Default, could be enhanced
                    speed_limit=50,  # Default
                    hour=hour,
                    day_of_week=day_of_week,
                    weather=weather,
                    lat=lat,
                    lon=lon
                )
                safety_scores.append(safety_result['score'])
            
            # Aggregate scores
            avg_risk = np.mean(risk_scores)
            max_risk = np.max(risk_scores)
            avg_safety = np.mean(safety_scores)
            min_safety = np.min(safety_scores)
            
            # Combined route score (weighted toward worst segments)
            route_safety_score = 0.6 * avg_safety + 0.4 * min_safety
            
            # Determine route category
            if route_safety_score >= 75:
                category = "Very Safe"
                color = "green"
            elif route_safety_score >= 60:
                category = "Safe"
                color = "lightgreen"
            elif route_safety_score >= 45:
                category = "Moderate"
                color = "yellow"
            elif route_safety_score >= 30:
                category = "Caution"
                color = "orange"
            else:
                category = "High Risk"
                color = "red"
            
            analyzed_routes.append({
                'route_id': route['route_id'],
                'distance_km': route['distance_km'],
                'duration_min': route['duration_min'],
                'geometry': geometry,
                'sampled_points': sampled_points,
                'safety_score': round(route_safety_score, 1),
                'average_risk': round(avg_risk * 100, 1),
                'max_risk_probability': round(max_risk * 100, 1),
                'category': category,
                'color': color,
                'risk_scores': [round(r * 100, 1) for r in risk_scores],
                'safety_scores': [round(s, 1) for s in safety_scores]
            })
        
        # Sort by safety score (highest = safest)
        analyzed_routes.sort(key=lambda x: x['safety_score'], reverse=True)
        
        # Add recommendations
        for i, route in enumerate(analyzed_routes):
            if i == 0:
                route['recommendation'] = "Safest Route ⭐"
            elif route['distance_km'] == min(r['distance_km'] for r in analyzed_routes):
                route['recommendation'] = "Shortest Route"
            elif route['duration_min'] == min(r['duration_min'] for r in analyzed_routes):
                route['recommendation'] = "Fastest Route"
            else:
                route['recommendation'] = f"Alternative {i}"
        
        return analyzed_routes
    
    def _predict_point_risk(self, lat: float, lon: float, hour: int,
                          day_of_week: str, weather: str) -> Dict:
        """
        Predict accident risk for a single point.
        
        Args:
            lat, lon: Coordinates
            hour: Hour of day
            day_of_week: Day name
            weather: Weather conditions
            
        Returns:
            Dictionary with risk prediction
        """
        # Create input dataframe for model
        is_night = 1 if (hour >= 18 or hour < 6) else 0
        
        input_data = {
            'hour': hour,
            'is_night': is_night,
            'location_accident_count': 0,  # Unknown for new point
            'Day_of_Week': day_of_week,
            'Weather_conditions': weather,
            'Road_surface_type': 'Single carriageway',
            'Speed_limit': 50,
            'Traffic Control Presence': 'None',
            'Area_accident_occured': 'Unknown',
            'Month': pd.to_datetime("today").month,
            'Year': pd.to_datetime("today").year,
            'Sex_of_driver': 'Male',
            'State Name': 'Karnataka',
            'City Name': 'Bangalore',
            'Alcohol Involvement': 'No',
            'Type_of_vehicle': 'Automobile',
            'Number_of_casualties': 1,
            'Age_band_of_driver': 25,
            'Driver License Status': 'Valid',
            'Road_surface_conditions': 'Dry' if 'Rain' not in weather else 'Wet or damp',
            'Light_conditions': 'Daylight' if not is_night else 'Darkness - lights lit',
            'Number_of_vehicles_involved': 2,
            'Number of Fatalities': 0
        }
        
        input_df = pd.DataFrame([input_data])
        
        try:
            # Predict
            probs = self.pipeline.predict_proba(input_df)[0]
            predicted_class = np.argmax(probs)
            
            risk_labels = ["Low", "Medium", "High"]
            
            return {
                'risk_level': risk_labels[predicted_class],
                'risk_probability': probs[predicted_class],
                'probabilities': {
                    'low': probs[0],
                    'medium': probs[1],
                    'high': probs[2]
                }
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'risk_level': 'Unknown',
                'risk_probability': 0.5,
                'probabilities': {'low': 0.33, 'medium': 0.34, 'high': 0.33}
            }
    
    def compare_routes(self, routes: List[Dict]) -> Dict:
        """
        Generate comparison summary for multiple routes.
        
        Args:
            routes: List of analyzed route dictionaries
            
        Returns:
            Comparison summary
        """
        if not routes:
            return {'error': 'No routes to compare'}
        
        safest = max(routes, key=lambda x: x['safety_score'])
        shortest = min(routes, key=lambda x: x['distance_km'])
        fastest = min(routes, key=lambda x: x['duration_min'])
        
        # Calculate trade-offs
        safety_distance_tradeoff = []
        for route in routes:
            tradeoff_score = (route['safety_score'] / 100) * 0.6 + \
                           (1 - route['distance_km'] / max(r['distance_km'] for r in routes)) * 0.4
            safety_distance_tradeoff.append({
                'route_id': route['route_id'],
                'tradeoff_score': round(tradeoff_score, 3)
            })
        
        best_tradeoff = max(safety_distance_tradeoff, key=lambda x: x['tradeoff_score'])
        
        return {
            'total_routes': len(routes),
            'safest_route': {
                'id': safest['route_id'],
                'safety_score': safest['safety_score'],
                'distance_km': safest['distance_km'],
                'duration_min': safest['duration_min']
            },
            'shortest_route': {
                'id': shortest['route_id'],
                'distance_km': shortest['distance_km'],
                'safety_score': shortest['safety_score']
            },
            'fastest_route': {
                'id': fastest['route_id'],
                'duration_min': fastest['duration_min'],
                'safety_score': fastest['safety_score']
            },
            'recommended_route': best_tradeoff['route_id'],
            'recommendation_reason': 'Best balance of safety and distance'
        }


if __name__ == "__main__":
    print("Route Analyzer module loaded")
