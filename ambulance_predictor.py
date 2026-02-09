"""
Ambulance Response Time Predictor

Estimates emergency response times:
- Finds nearest hospitals using OpenStreetMap
- Calculates route distances and travel times
- Estimates total response time (preparation + travel)
- Ranks hospitals by ETA
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from geospatial_utils import find_nearby_hospitals, get_route_distance_time


class AmbulanceResponsePredictor:
    """Predict ambulance response times to accident locations."""
    
    # Default parameters (can be customized)
    PREPARATION_TIME_MIN = 5  # Average dispatch + ambulance ready time
    PREPARATION_TIME_MAX = 8
    
    # Average speeds by road type (km/h)
    SPEED_BY_ROAD_TYPE = {
        'highway': 80,
        'urban': 40,
        'residential': 30
    }
    
    # Time-of-day speed multipliers
    TIME_SPEED_MULTIPLIERS = {
        'night': 1.2,      # Faster at night (less traffic)
        'rush_hour': 0.7,  # Slower during rush hour
        'normal': 1.0
    }
    
    def __init__(self):
        pass
    
    def predict_response_times(self, accident_lat: float, accident_lon: float,
                              hour: int = 12, max_hospitals: int = 5,
                              radius_km: float = 25) -> List[Dict]:
        """
        Predict ambulance response times from nearby hospitals.
        
        Args:
            accident_lat, accident_lon: Accident location coordinates
            hour: Hour of day (0-23) for traffic adjustment
            max_hospitals: Maximum number of hospitals to analyze
            radius_km: Search radius for hospitals
            
        Returns:
            List of hospital response dictionaries sorted by ETA
        """
        # Find nearby hospitals
        hospitals = find_nearby_hospitals(accident_lat, accident_lon, radius_km)
        
        if not hospitals:
            return []
        
        # Limit to top N closest hospitals
        hospitals = hospitals[:max_hospitals]
        
        # Calculate response times for each hospital
        response_predictions = []
        
        for hospital in hospitals:
            # Get detailed route information
            route_info = get_route_distance_time(
                hospital['lat'], hospital['lon'],
                accident_lat, accident_lon
            )
            
            # Calculate preparation time (randomized within range)
            prep_time = np.random.uniform(
                self.PREPARATION_TIME_MIN,
                self.PREPARATION_TIME_MAX
            )
            
            # Adjust travel time based on conditions
            base_travel_time = route_info['duration_min']
            adjusted_travel_time = self._adjust_travel_time(base_travel_time, hour)
            
            # Total ETA
            total_eta = prep_time + adjusted_travel_time
            
            # Categorize response time
            if total_eta <= 8:
                response_category = "Excellent"
                priority = 1
            elif total_eta <= 12:
                response_category = "Good"
                priority = 2
            elif total_eta <= 20:
                response_category = "Acceptable"
                priority = 3
            else:
                response_category = "Delayed"
                priority = 4
            
            response_predictions.append({
                'hospital_name': hospital['name'],
                'hospital_lat': hospital['lat'],
                'hospital_lon': hospital['lon'],
                'distance_km': route_info['distance_km'],
                'straight_line_distance': hospital['distance_km'],
                'has_emergency': hospital.get('emergency', False),
                'preparation_time_min': round(prep_time, 1),
                'travel_time_min': round(adjusted_travel_time, 1),
                'total_eta_min': round(total_eta, 1),
                'response_category': response_category,
                'priority': priority,
                'route_geometry': route_info.get('geometry', [])
            })
        
        # Sort by total ETA (fastest first)
        response_predictions.sort(key=lambda x: x['total_eta_min'])
        
        # Add rankings
        for rank, pred in enumerate(response_predictions, 1):
            pred['rank'] = rank
            if rank == 1:
                pred['recommendation'] = "Fastest Response ⚡"
            elif pred['has_emergency']:
                pred['recommendation'] = "Emergency Facility +"
            else:
                pred['recommendation'] = f"Option {rank}"
        
        return response_predictions
    
    def _adjust_travel_time(self, base_time_min: float, hour: int) -> float:
        """
        Adjust travel time based on time of day.
        
        Args:
            base_time_min: Base travel time from routing API
            hour: Hour of day (0-23)
            
        Returns:
            Adjusted travel time in minutes
        """
        # Determine time period
        if hour in [8, 9, 17, 18, 19]:
            # Rush hour
            multiplier = self.TIME_SPEED_MULTIPLIERS['rush_hour']
        elif hour >= 22 or hour < 6:
            # Night time
            multiplier = self.TIME_SPEED_MULTIPLIERS['night']
        else:
            # Normal hours
            multiplier = self.TIME_SPEED_MULTIPLIERS['normal']
        
        # Ambulances have priority, so reduce impact of traffic
        # Emergency vehicles can bypass some congestion
        emergency_factor = 0.8  # 20% reduction in delay
        
        adjusted_time = base_time_min * (1 + (multiplier - 1) * emergency_factor)
        
        return adjusted_time
    
    def get_optimal_hospital(self, accident_lat: float, accident_lon: float,
                           hour: int = 12, consider_facilities: bool = True) -> Optional[Dict]:
        """
        Get the single optimal hospital recommendation.
        
        Args:
            accident_lat, accident_lon: Accident coordinates
            hour: Hour of day
            consider_facilities: Whether to prefer emergency facilities
            
        Returns:
            Best hospital dictionary or None
        """
        predictions = self.predict_response_times(accident_lat, accident_lon, hour)
        
        if not predictions:
            return None
        
        # If considering facilities, prefer emergency-capable hospitals
        if consider_facilities:
            emergency_hospitals = [p for p in predictions if p['has_emergency']]
            if emergency_hospitals:
                # Among emergency hospitals, pick fastest
                return emergency_hospitals[0]
        
        # Otherwise, return fastest overall
        return predictions[0]
    
    def estimate_survival_probability(self, eta_minutes: float) -> Dict:
        """
        Estimate survival probability based on response time.
        (Based on medical research for the "Golden Hour")
        
        Args:
            eta_minutes: Estimated time to arrival
            
        Returns:
            Dictionary with survival estimates
        """
        # Golden hour concept: Critical first 60 minutes
        if eta_minutes <= 10:
            survival_rate = 0.95
            severity_impact = "Minimal"
        elif eta_minutes <= 20:
            survival_rate = 0.85
            severity_impact = "Low"
        elif eta_minutes <= 40:
            survival_rate = 0.70
            severity_impact = "Moderate"
        elif eta_minutes <= 60:
            survival_rate = 0.55
            severity_impact = "Significant"
        else:
            survival_rate = 0.40
            severity_impact = "Critical"
        
        return {
            'estimated_survival_rate': round(survival_rate, 2),
            'severity_impact': severity_impact,
            'within_golden_hour': eta_minutes <= 60,
            'urgency': 'Critical' if eta_minutes > 20 else 'High' if eta_minutes > 10 else 'Normal'
        }
    
    def generate_response_summary(self, predictions: List[Dict]) -> str:
        """
        Generate human-readable summary of response predictions.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Formatted summary string
        """
        if not predictions:
            return "No hospitals found in the area."
        
        best = predictions[0]
        
        summary_lines = []
        summary_lines.append(f"🚑 **Fastest Response**: {best['hospital_name']}")
        summary_lines.append(f"   • Distance: {best['distance_km']} km")
        summary_lines.append(f"   • Estimated ETA: {best['total_eta_min']} minutes")
        summary_lines.append(f"   • Response Quality: {best['response_category']}")
        
        survival = self.estimate_survival_probability(best['total_eta_min'])
        summary_lines.append(f"   • Urgency Level: {survival['urgency']}")
        
        if len(predictions) > 1:
            summary_lines.append(f"\n📍 {len(predictions)} hospitals identified within range")
        
        return "\n".join(summary_lines)


if __name__ == "__main__":
    print("Ambulance Response Predictor module loaded")
    
    # Test
    predictor = AmbulanceResponsePredictor()
    
    # Example: Accident in Bangalore
    results = predictor.predict_response_times(12.9716, 77.5946, hour=20)
    for r in results[:3]:
        print(f"{r['hospital_name']}: ETA {r['total_eta_min']} min ({r['response_category']})")
