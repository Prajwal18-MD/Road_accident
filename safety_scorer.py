"""
Road Safety Scoring System

Calculates safety scores (0-100) for locations and routes based on:
- Historical accident data
- Road conditions (type, speed limits)
- Temporal factors (time of day, day of week)
- Weather conditions
- Real-time adjustments
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os


class SafetyScorer:
    """Calculate and manage road safety scores."""
    
    def __init__(self, accident_data_path: Optional[str] = None):
        """
        Initialize safety scorer.
        
        Args:
            accident_data_path: Path to historical accident CSV data
        """
        self.accident_data = None
        self.location_stats = {}
        self.area_risk_map = {}
        
        if accident_data_path and os.path.exists(accident_data_path):
            self._load_accident_data(accident_data_path)
    
    def _load_accident_data(self, data_path: str):
        """Load and preprocess accident data."""
        try:
            self.accident_data = pd.read_csv(data_path)
            
            # Calculate statistics per location
            if 'Area_accident_occured' in self.accident_data.columns:
                location_counts = self.accident_data['Area_accident_occured'].value_counts()
                self.location_stats = location_counts.to_dict()
                
                # Calculate risk scores per area (normalized)
                max_accidents = location_counts.max()
                for area, count in self.location_stats.items():
                    # Higher accident count = lower safety score
                    risk_factor = count / max_accidents if max_accidents > 0 else 0
                    self.area_risk_map[area] = risk_factor
                    
            print(f"Loaded {len(self.accident_data)} accident records")
            print(f"Tracked {len(self.location_stats)} unique locations")
        except Exception as e:
            print(f"Error loading accident data: {e}")
    
    def calculate_base_score(self, area: str, lat: float = None, lon: float = None) -> float:
        """
        Calculate base safety score from historical data.
        
        Args:
            area: Area/road name
            lat, lon: Optional coordinates for proximity-based scoring
            
        Returns:
            Base safety score (0-100)
        """
        base_score = 100.0
        
        if area in self.area_risk_map:
            risk_factor = self.area_risk_map[area]
            # Deduct up to 40 points based on accident history
            deduction = risk_factor * 40
            base_score -= deduction
        elif area in self.location_stats:
            # Have accident count but not in risk map
            count = self.location_stats[area]
            # Logarithmic penalty for accident count
            deduction = min(40, np.log1p(count) * 5)
            base_score -= deduction
        
        return max(0, base_score)
    
    def adjust_for_road_conditions(self, score: float, road_type: str, 
                                   speed_limit: int, traffic_level: str = "Medium") -> float:
        """
        Adjust score based on road conditions.
        
        Args:
            score: Current safety score
            road_type: Type of road
            speed_limit: Speed limit in km/h
            traffic_level: Current traffic level (Low/Medium/High)
            
        Returns:
            Adjusted safety score
        """
        # Road type risk factors
        road_risk = {
            "Highway": -10,
            "Single carriageway": -5,
            "Dual carriageway": 0,
            "Roundabout": -3,
            "One way": 2
        }
        score += road_risk.get(road_type, 0)
        
        # Speed limit penalty (higher speed = higher risk)
        if speed_limit >= 100:
            score -= 15
        elif speed_limit >= 80:
            score -= 10
        elif speed_limit >= 60:
            score -= 5
        else:
            score += 5  # Low speed zones are safer
        
        # Traffic level adjustment
        traffic_adjustment = {
            "Low": 5,      # Less traffic = safer
            "Medium": 0,
            "High": -8     # Congestion increases risk
        }
        score += traffic_adjustment.get(traffic_level, 0)
        
        return max(0, min(100, score))
    
    def adjust_for_time(self, score: float, hour: int, day_of_week: str, 
                        is_night: bool = False) -> float:
        """
        Adjust score based on temporal factors.
        
        Args:
            score: Current safety score
            hour: Hour of day (0-23)
            day_of_week: Day name
            is_night: Whether it's nighttime
            
        Returns:
            Adjusted safety score
        """
        # Night time penalty
        if is_night or hour >= 22 or hour < 6:
            score -= 12
        
        # Rush hour penalty (reduced visibility, stress)
        if hour in [8, 9, 17, 18, 19]:
            score -= 5
        
        # Late night high risk
        if 0 <= hour < 4:
            score -= 8  # Additional penalty for very late night
        
        # Weekend adjustment (different traffic patterns)
        if day_of_week in ["Saturday", "Sunday"]:
            score -= 3  # Slightly higher risk on weekends (leisure driving)
        
        return max(0, min(100, score))
    
    def adjust_for_weather(self, score: float, weather: str) -> float:
        """
        Adjust score based on weather conditions.
        
        Args:
            score: Current safety score
            weather: Weather condition
            
        Returns:
            Adjusted safety score
        """
        weather_risk = {
            "Fine": 0,
            "Raining": -15,
            "Raining and Windy": -25,
            "Fog or Mist": -20,
            "Snowing": -30,
            "Other": -10
        }
        
        adjustment = weather_risk.get(weather, -10)
        score += adjustment
        
        return max(0, min(100, score))
    
    def calculate_safety_score(self, area: str, road_type: str, speed_limit: int,
                              hour: int, day_of_week: str, weather: str,
                              traffic_level: str = "Medium", 
                              lat: float = None, lon: float = None) -> Dict:
        """
        Calculate comprehensive safety score with all factors.
        
        Args:
            area: Location/area name
            road_type: Type of road
            speed_limit: Speed limit in km/h
            hour: Hour of day (0-23)
            day_of_week: Day name
            weather: Weather condition
            traffic_level: Traffic density
            lat, lon: Optional coordinates
            
        Returns:
            Dictionary with score, category, and breakdown
        """
        # Start with base score
        score = self.calculate_base_score(area, lat, lon)
        base = score
        
        # Apply adjustments
        is_night = hour >= 18 or hour < 6
        
        score = self.adjust_for_road_conditions(score, road_type, speed_limit, traffic_level)
        road_adjusted = score
        
        score = self.adjust_for_time(score, hour, day_of_week, is_night)
        time_adjusted = score
        
        score = self.adjust_for_weather(score, weather)
        final_score = max(0, min(100, score))
        
        # Categorize score
        if final_score >= 80:
            category = "Very Safe"
            color = "green"
        elif final_score >= 60:
            category = "Moderate"
            color = "yellow"
        elif final_score >= 40:
            category = "Caution"
            color = "orange"
        else:
            category = "High Risk"
            color = "red"
        
        return {
            'score': round(final_score, 1),
            'category': category,
            'color': color,
            'breakdown': {
                'base_score': round(base, 1),
                'after_road_conditions': round(road_adjusted, 1),
                'after_time_factors': round(time_adjusted, 1),
                'final_score': round(final_score, 1)
            },
            'factors': {
                'historical_risk': round(100 - base, 1),
                'road_conditions': round(road_adjusted - base, 1),
                'time_factors': round(time_adjusted - road_adjusted, 1),
                'weather_impact': round(final_score - time_adjusted, 1)
            }
        }
    
    def calculate_route_safety_score(self, route_points: List[Tuple[float, float]], 
                                     road_types: List[str] = None,
                                     speed_limits: List[int] = None,
                                     hour: int = 12, day_of_week: str = "Monday",
                                     weather: str = "Fine") -> Dict:
        """
        Calculate aggregate safety score for an entire route.
        
        Args:
            route_points: List of (lat, lon) coordinate tuples along route
            road_types: List of road types for each segment (or None for default)
            speed_limits: List of speed limits for each segment (or None for default)
            hour, day_of_week, weather: Temporal and weather conditions
            
        Returns:
            Route safety score dictionary
        """
        if not route_points:
            return {'score': 0, 'category': 'Unknown'}
        
        scores = []
        for i, (lat, lon) in enumerate(route_points):
            road_type = road_types[i] if road_types and i < len(road_types) else "Single carriageway"
            speed_limit = speed_limits[i] if speed_limits and i < len(speed_limits) else 50
            
            # For route scoring, use generic area (no specific location data)
            segment_score = self.calculate_safety_score(
                area="Unknown",
                road_type=road_type,
                speed_limit=speed_limit,
                hour=hour,
                day_of_week=day_of_week,
                weather=weather,
                lat=lat,
                lon=lon
            )
            scores.append(segment_score['score'])
        
        # Aggregate: use weighted average (can be customized)
        avg_score = np.mean(scores)
        min_score = np.min(scores)
        
        # Final route score: weighted toward minimum (weakest link matters)
        route_score = 0.7 * avg_score + 0.3 * min_score
        
        if route_score >= 80:
            category = "Very Safe"
        elif route_score >= 60:
            category = "Moderate"
        elif route_score >= 40:
            category = "Caution"
        else:
            category = "High Risk"
        
        return {
            'score': round(route_score, 1),
            'average_score': round(avg_score, 1),
            'minimum_score': round(min_score, 1),
            'category': category,
            'segment_count': len(scores)
        }


if __name__ == "__main__":
    # Test safety scorer
    print("Testing Safety Scorer...")
    
    # Initialize with mock data
    scorer = SafetyScorer()
    
    # Test single location score
    result = scorer.calculate_safety_score(
        area="MG Road",
        road_type="Single carriageway",
        speed_limit=50,
        hour=20,  # 8 PM
        day_of_week="Friday",
        weather="Raining",
        traffic_level="High"
    )
    
    print(f"\nSafety Score: {result['score']}/100")
    print(f"Category: {result['category']}")
    print(f"Breakdown: {result['breakdown']}")
    print(f"Factors: {result['factors']}")
