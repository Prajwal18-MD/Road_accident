"""
AI-Driven Safety Recommendations

Provides intelligent safety improvement suggestions:
- Pattern detection from accident clusters
- Infrastructure improvement recommendations
- Traffic optimization suggestions
- Priority ranking by predicted impact
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import json


class SafetyRecommendationEngine:
    """Generate AI-driven safety improvement recommendations."""
    
    def __init__(self, accident_data: pd.DataFrame = None):
        """
        Initialize recommendation engine.
        
        Args:
            accident_data: Historical accident DataFrame
        """
        self.accident_data = accident_data
        self.patterns = {}
        
        if accident_data is not None:
            self._analyze_patterns()
    
    def _analyze_patterns(self):
        """Analyze accident data for patterns."""
        if self.accident_data is None or len(self.accident_data) == 0:
            return
        
        df = self.accident_data
        
        # Time-based patterns
        if 'Time' in df.columns:
            try:
                df['Time'] = pd.to_datetime(df['Time'], format='mixed', errors='coerce')
                df['hour'] = df['Time'].dt.hour
                hourly_accidents = df['hour'].value_counts().to_dict()
                self.patterns['peak_hours'] = sorted(
                    hourly_accidents.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            except:
                pass
        
        # Location hotspots
        if 'Area_accident_occured' in df.columns:
            location_counts = df['Area_accident_occured'].value_counts()
            self.patterns['hotspot_locations'] = location_counts.head(10).to_dict()
        
        # Weather-related
        if 'Weather_conditions' in df.columns:
            weather_accidents = df['Weather_conditions'].value_counts().to_dict()
            self.patterns['weather_impact'] = weather_accidents
        
        # Road type analysis
        if 'Road_surface_type' in df.columns:
            road_accidents = df['Road_surface_type'].value_counts().to_dict()
            self.patterns['road_type_risk'] = road_accidents
        
        print(f"Patterns analyzed from {len(df)} accident records")
    
    def get_location_recommendations(self, area: str, lat: float = None, 
                                    lon: float = None, accident_count: int = 0) -> List[Dict]:
        """
        Get safety recommendations for a specific location.
        
        Args:
            area: Location/area name
            lat, lon: Coordinates (optional)
            accident_count: Historical accident count at this location
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # High accident area recommendations
        if accident_count > 10:
            recommendations.append({
                'category': 'Infrastructure',
                'priority': 'High',
                'action': 'Install Speed Breakers',
                'description': f'Area has {accident_count} recorded accidents. Speed breakers can reduce accident severity by 40%.',
                'estimated_impact': 'High',
                'implementation_time': 'Short-term (1-2 months)',
                'cost': 'Low'
            })
            
            recommendations.append({
                'category': 'Enforcement',
                'priority': 'High',
                'action': 'Deploy Traffic Police',
                'description': 'Regular police presence during peak hours to enforce speed limits and traffic rules.',
                'estimated_impact': 'High',
                'implementation_time': 'Immediate',
                'cost': 'Medium'
            })
        
        if accident_count > 5:
            recommendations.append({
                'category': 'Infrastructure',
                'priority': 'Medium',
                'action': 'Improve Street Lighting',
                'description': 'Enhanced lighting reduces nighttime accidents by up to 30%.',
                'estimated_impact': 'Medium',
                'implementation_time': 'Medium-term (2-4 months)',
                'cost': 'Medium'
            })
            
            recommendations.append({
                'category': 'Signage',
                'priority': 'Medium',
                'action': 'Install Warning Signs',
                'description': 'Place "Accident Prone Zone" and speed limit signs 500m before the area.',
                'estimated_impact': 'Medium',
                'implementation_time': 'Short-term',
                'cost': 'Low'
            })
        
        # Check patterns for time-specific recommendations
        if hasattr(self, 'patterns') and 'peak_hours' in self.patterns:
            peak_times = [hour for hour, count in self.patterns['peak_hours'][:3]]
            
            recommendations.append({
                'category': 'Traffic Management',
                'priority': 'Medium',
                'action': 'Traffic Signal Optimization',
                'description': f'Peak accident hours: {peak_times}. Optimize signal timing during these periods.',
                'estimated_impact': 'Medium',
                'implementation_time': 'Short-term',
                'cost': 'Low'
            })
        
        return recommendations
    
    def get_weather_based_recommendations(self, weather: str, area: str = None) -> List[Dict]:
        """
        Get recommendations based on weather conditions.
        
        Args:
            weather: Current weather condition
            area: Optional location
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if 'Rain' in weather:
            recommendations.append({
                'category': 'Road Maintenance',
                'priority': 'High',
                'action': 'Improve Drainage System',
                'description': 'Water accumulation increases accident risk. Ensure proper drainage.',
                'estimated_impact': 'High',
                'implementation_time': 'Long-term (6-12 months)',
                'cost': 'High'
            })
            
            recommendations.append({
                'category': 'Road Surface',
                'priority': 'Medium',
                'action': 'Apply Anti-Skid Surface',
                'description': 'Special road coating to improve tire grip in wet conditions.',
                'estimated_impact': 'Medium',
                'implementation_time': 'Medium-term',
                'cost': 'Medium'
            })
            
            recommendations.append({
                'category': 'Communication',
                'priority': 'Immediate',
                'action': 'Activate Weather Alerts',
                'description': 'Send SMS/app notifications warning drivers about wet road conditions.',
                'estimated_impact': 'Low',
                'implementation_time': 'Immediate',
                'cost': 'Low'
            })
        
        if 'Fog' in weather or 'Mist' in weather:
            recommendations.append({
                'category': 'Infrastructure',
                'priority': 'High',
                'action': 'Install Reflective Road Markers',
                'description': 'Reflective markers improve visibility in fog by 50%.',
                'estimated_impact': 'High',
                'implementation_time': 'Short-term',
                'cost': 'Low'
            })
        
        return recommendations
    
    def get_time_based_recommendations(self, hour: int, is_night: bool) -> List[Dict]:
        """
        Get recommendations based on time of day.
        
        Args:
            hour: Hour of day (0-23)
            is_night: Whether it's nighttime
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if is_night:
            recommendations.append({
                'category': 'Enforcement',
                'priority': 'High',
                'action': 'Nighttime Speed Enforcement',
                'description': 'Deploy speed cameras and increase night patrols. Night driving has 3x higher fatality rate.',
                'estimated_impact': 'High',
                'implementation_time': 'Immediate',
                'cost': 'Medium'
            })
            
            recommendations.append({
                'category': 'Driver Awareness',
                'priority': 'Medium',
                'action': 'Fatigue Detection Campaign',
                'description': 'Educate drivers about fatigue risks. Install rest areas every 50 km on highways.',
                'estimated_impact': 'Medium',
                'implementation_time': 'Medium-term',
                'cost': 'Medium'
            })
        
        # Rush hour recommendations
        if hour in [8, 9, 17, 18, 19]:
            recommendations.append({
                'category': 'Traffic Management',
                'priority': 'High',
                'action': 'Dedicated Lane Management',
                'description': 'Implement reversible lanes or bus-only lanes during peak hours.',
                'estimated_impact': 'Medium',
                'implementation_time': 'Long-term',
                'cost': 'High'
            })
        
        return recommendations
    
    def get_road_improvement_recommendations(self, road_type: str, speed_limit: int) -> List[Dict]:
        """
        Get recommendations based on road characteristics.
        
        Args:
            road_type: Type of road
            speed_limit: Speed limit in km/h
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if speed_limit >= 80:
            recommendations.append({
                'category': 'Infrastructure',
                'priority': 'High',
                'action': 'Install Median Barriers',
                'description': 'High-speed roads need median barriers to prevent head-on collisions.',
                'estimated_impact': 'Very High',
                'implementation_time': 'Long-term',
                'cost': 'High'
            })
        
        if 'Single carriageway' in road_type:
            recommendations.append({
                'category': 'Road Design',
                'priority': 'Medium',
                'action': 'Widen Road / Add Shoulder',
                'description': 'Single carriageways benefit from wider shoulders for emergency stops.',
                'estimated_impact': 'Medium',
                'implementation_time': 'Long-term',
                'cost': 'Very High'
            })
        
        if 'Roundabout' in road_type:
            recommendations.append({
                'category': 'Signage',
                'priority': 'Medium',
                'action': 'Improve Roundabout Marking',
                'description': 'Clear lane markings and yield signs reduce roundabout accidents by 35%.',
                'estimated_impact': 'Medium',
                'implementation_time': 'Short-term',
                'cost': 'Low'
            })
        
        return recommendations
    
    def generate_comprehensive_recommendations(self, context: Dict) -> Dict:
        """
        Generate comprehensive recommendations based on full context.
        
        Args:
            context: Dictionary with keys like area, hour, weather, road_type, etc.
            
        Returns:
            Comprehensive recommendation package
        """
        all_recommendations = []
        
        # Location-based
        if 'area' in context:
            all_recommendations.extend(
                self.get_location_recommendations(
                    context['area'],
                    context.get('lat'),
                    context.get('lon'),
                    context.get('accident_count', 0)
                )
            )
        
        # Weather-based
        if 'weather' in context:
            all_recommendations.extend(
                self.get_weather_based_recommendations(
                    context['weather'],
                    context.get('area')
                )
            )
        
        # Time-based
        if 'hour' in context:
            is_night = context.get('is_night', context['hour'] >= 18 or context['hour'] < 6)
            all_recommendations.extend(
                self.get_time_based_recommendations(context['hour'], is_night)
            )
        
        # Road-based
        if 'road_type' in context and 'speed_limit' in context:
            all_recommendations.extend(
                self.get_road_improvement_recommendations(
                    context['road_type'],
                    context['speed_limit']
                )
            )
        
        # Remove duplicates and prioritize
        unique_recs = []
        seen_actions = set()
        
        for rec in all_recommendations:
            if rec['action'] not in seen_actions:
                unique_recs.append(rec)
                seen_actions.add(rec['action'])
        
        # Sort by priority
        priority_order = {'High': 1, 'Medium': 2, 'Low': 3, 'Immediate': 0}
        unique_recs.sort(key=lambda x: priority_order.get(x['priority'], 4))
        
        # Categorize
        by_category = defaultdict(list)
        for rec in unique_recs:
            by_category[rec['category']].append(rec)
        
        return {
            'total_recommendations': len(unique_recs),
            'all_recommendations': unique_recs,
            'by_category': dict(by_category),
            'high_priority': [r for r in unique_recs if r['priority'] in ['High', 'Immediate']],
            'quick_wins': [r for r in unique_recs if r['implementation_time'] in ['Immediate', 'Short-term (1-2 months)', 'Short-term']],
        }
    
    def estimate_impact(self, recommendations: List[Dict], current_accident_rate: float) -> Dict:
        """
        Estimate the potential impact of implementing recommendations.
        
        Args:
            recommendations: List of recommendation dictionaries
            current_accident_rate: Current accident rate (accidents per month/year)
            
        Returns:
            Impact estimation
        """
        impact_multipliers = {
            'Very High': 0.4,  # 40% reduction
            'High': 0.25,      # 25% reduction
            'Medium': 0.15,    # 15% reduction
            'Low': 0.05        # 5% reduction
        }
        
        total_reduction = 0
        for rec in recommendations:
            impact = rec.get('estimated _impact', 'Low')
            total_reduction += impact_multipliers.get(impact, 0.05)
        
        # Cap at 70% maximum reduction (realistic)
        total_reduction = min(total_reduction, 0.70)
        
        projected_rate = current_accident_rate * (1 - total_reduction)
        accidents_prevented = current_accident_rate - projected_rate
        
        return {
            'current_accident_rate': current_accident_rate,
            'projected_accident_rate': round(projected_rate, 2),
            'estimated_reduction_percent': round(total_reduction * 100, 1),
            'accidents_prevented_per_period': round(accidents_prevented, 1),
            'recommendations_count': len(recommendations)
        }


if __name__ == "__main__":
    print("Safety Recommendation Engine loaded")
    
    # Test
    engine = SafetyRecommendationEngine()
    
    context = {
        'area': 'MG Road',
        'accident_count': 15,
        'hour': 20,
        'is_night': True,
        'weather': 'Raining',
        'road_type': 'Single carriageway',
        'speed_limit': 60
    }
    
    results = engine.generate_comprehensive_recommendations(context)
    print(f"\nGenerated {results['total_recommendations']} recommendations")
    print(f"High priority: {len(results['high_priority'])}")
    print(f"Quick wins: {len(results['quick_wins'])}")
