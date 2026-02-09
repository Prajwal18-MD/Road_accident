"""
Geospatial Utilities for Road Safety System

Provides functions for:
- Geocoding (address <-> coordinates)
- Finding nearby hospitals and emergency services
- Route distance calculations
- Location clustering for hotspot detection
"""

import requests
import time
from typing import List, Dict, Tuple, Optional
from math import radians, sin, cos, sqrt, atan2
import json

# OpenStreetMap Nominatim API (free, rate-limited to 1 req/sec)
NOMINATIM_URL = "https://nominatim.openstreetmap.org"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OSRM_URL = "http://router.project-osrm.org"

# Cache for geocoding results to avoid repeated API calls
_geocoding_cache = {}
_hospital_cache = {}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
        
    Returns:
        Distance in kilometers
    """
    R = 6371.0  # Earth radius in kilometers
    
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    return distance


def geocode_address(address: str, use_cache: bool = True) -> Optional[Tuple[float, float]]:
    """
    Convert address to coordinates using Nominatim API.
    
    Args:
        address: Street address or place name
        use_cache: Whether to use cached results
        
    Returns:
        (latitude, longitude) tuple or None if not found
    """
    if use_cache and address in _geocoding_cache:
        return _geocoding_cache[address]
    
    try:
        params = {
            'q': address,
            'format': 'json',
            'limit': 1
        }
        headers = {'User-Agent': 'RoadSafetyApp/1.0'}
        
        response = requests.get(f"{NOMINATIM_URL}/search", params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            _geocoding_cache[address] = (lat, lon)
            time.sleep(1)  # Rate limit: 1 request per second
            return (lat, lon)
    except Exception as e:
        print(f"Geocoding error for '{address}': {e}")
    
    return None


def reverse_geocode(lat: float, lon: float, use_cache: bool = True) -> Optional[str]:
    """
    Convert coordinates to address using Nominatim API.
    
    Args:
        lat, lon: Coordinates
        use_cache: Whether to use cached results
        
    Returns:
        Address string or None if not found
    """
    cache_key = f"{lat:.4f},{lon:.4f}"
    if use_cache and cache_key in _geocoding_cache:
        return _geocoding_cache[cache_key]
    
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json'
        }
        headers = {'User-Agent': 'RoadSafetyApp/1.0'}
        
        response = requests.get(f"{NOMINATIM_URL}/reverse", params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if 'display_name' in data:
            address = data['display_name']
            _geocoding_cache[cache_key] = address
            time.sleep(1)  # Rate limit
            return address
    except Exception as e:
        print(f"Reverse geocoding error for ({lat}, {lon}): {e}")
    
    return None


def find_nearby_hospitals(lat: float, lon: float, radius_km: float = 20, 
                          use_cache: bool = True) -> List[Dict]:
    """
    Find nearby hospitals using OpenStreetMap Overpass API.
    
    Args:
        lat, lon: Center point coordinates
        radius_km: Search radius in kilometers
        use_cache: Whether to use cached results
        
    Returns:
        List of hospital dictionaries with keys: name, lat, lon, distance_km, emergency
    """
    cache_key = f"{lat:.3f},{lon:.3f},{radius_km}"
    if use_cache and cache_key in _hospital_cache:
        return _hospital_cache[cache_key]
    
    radius_m = radius_km * 1000
    
    # Overpass QL query for hospitals and emergency facilities
    overpass_query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      way["amenity"="hospital"](around:{radius_m},{lat},{lon});
      node["amenity"="clinic"]["emergency"="yes"](around:{radius_m},{lat},{lon});
    );
    out center;
    """
    
    try:
        response = requests.post(OVERPASS_URL, data={'data': overpass_query}, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        hospitals = []
        for element in data.get('elements', []):
            if element['type'] == 'node':
                h_lat, h_lon = element['lat'], element['lon']
            elif element['type'] == 'way' and 'center' in element:
                h_lat, h_lon = element['center']['lat'], element['center']['lon']
            else:
                continue
            
            name = element.get('tags', {}).get('name', 'Unnamed Hospital')
            emergency = element.get('tags', {}).get('emergency', 'no') == 'yes'
            
            distance = haversine_distance(lat, lon, h_lat, h_lon)
            
            hospitals.append({
                'name': name,
                'lat': h_lat,
                'lon': h_lon,
                'distance_km': round(distance, 2),
                'emergency': emergency
            })
        
        # Sort by distance
        hospitals.sort(key=lambda x: x['distance_km'])
        
        _hospital_cache[cache_key] = hospitals
        return hospitals
        
    except Exception as e:
        print(f"Error finding hospitals: {e}")
        # Return mock data for demo if API fails
        return _get_mock_hospitals(lat, lon)


def _get_mock_hospitals(lat: float, lon: float) -> List[Dict]:
    """Generate mock hospital data when API is unavailable."""
    return [
        {
            'name': 'City General Hospital',
            'lat': lat + 0.01,
            'lon': lon + 0.01,
            'distance_km': 1.5,
            'emergency': True
        },
        {
            'name': 'Community Medical Center',
            'lat': lat - 0.02,
            'lon': lon + 0.015,
            'distance_km': 2.8,
            'emergency': True
        },
        {
            'name': 'Regional Trauma Center',
            'lat': lat + 0.03,
            'lon': lon - 0.02,
            'distance_km': 4.2,
            'emergency': True
        }
    ]


def get_route_distance_time(start_lat: float, start_lon: float, 
                           end_lat: float, end_lon: float) -> Dict:
    """
    Get route distance and estimated travel time using OSRM API.
    
    Args:
        start_lat, start_lon: Starting coordinates
        end_lat, end_lon: Destination coordinates
        
    Returns:
        Dict with keys: distance_km, duration_min, geometry (route coordinates)
    """
    try:
        # OSRM expects lon,lat order (not lat,lon!)
        url = f"{OSRM_URL}/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
        params = {
            'overview': 'simplified',
            'geometries': 'geojson'
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data['code'] == 'Ok' and data['routes']:
            route = data['routes'][0]
            return {
                'distance_km': round(route['distance'] / 1000, 2),
                'duration_min': round(route['duration'] / 60, 1),
                'geometry': route['geometry']['coordinates']
            }
    except Exception as e:
        print(f"OSRM routing error: {e}")
    
    # Fallback to haversine approximation
    distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    # Assume average speed of 40 km/h in urban areas
    duration = (distance / 40) * 60
    
    return {
        'distance_km': round(distance, 2),
        'duration_min': round(duration, 1),
        'geometry': [[start_lon, start_lat], [end_lon, end_lat]]
    }


def get_alternative_routes(start_lat: float, start_lon: float, 
                          end_lat: float, end_lon: float, 
                          num_alternatives: int = 2) -> List[Dict]:
    """
    Get multiple route options between two points.
    
    Args:
        start_lat, start_lon: Starting coordinates
        end_lat, end_lon: Destination coordinates
        num_alternatives: Number of alternative routes (max 3)
        
    Returns:
        List of route dictionaries
    """
    try:
        url = f"{OSRM_URL}/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
        params = {
            'alternatives': min(num_alternatives, 3),
            'overview': 'simplified',
            'geometries': 'geojson',
            'steps': 'false'
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data['code'] == 'Ok':
            routes = []
            for idx, route in enumerate(data['routes']):
                routes.append({
                    'route_id': idx,
                    'distance_km': round(route['distance'] / 1000, 2),
                    'duration_min': round(route['duration'] / 60, 1),
                    'geometry': route['geometry']['coordinates']
                })
            return routes
    except Exception as e:
        print(f"Error getting alternative routes: {e}")
    
    # Fallback: return single route
    main_route = get_route_distance_time(start_lat, start_lon, end_lat, end_lon)
    return [{**main_route, 'route_id': 0}]


def sample_points_along_route(geometry: List[List[float]], num_points: int = 10) -> List[Tuple[float, float]]:
    """
    Sample evenly spaced points along a route geometry.
    
    Args:
        geometry: List of [lon, lat] coordinate pairs
        num_points: Number of points to sample
        
    Returns:
        List of (lat, lon) tuples
    """
    if len(geometry) <= num_points:
        return [(lon_lat[1], lon_lat[0]) for lon_lat in geometry]
    
    # Sample every nth point
    step = len(geometry) / num_points
    sampled = []
    for i in range(num_points):
        idx = int(i * step)
        lon_lat = geometry[idx]
        sampled.append((lon_lat[1], lon_lat[0]))  # Convert to (lat, lon)
    
    return sampled


if __name__ == "__main__":
    # Test functions
    print("Testing Geospatial Utils...")
    
    # Test haversine distance
    bangalore_lat, bangalore_lon = 12.9716, 77.5946
    mumbai_lat, mumbai_lon = 19.0760, 72.8777
    
    dist = haversine_distance(bangalore_lat, bangalore_lon, mumbai_lat, mumbai_lon)
    print(f"Bangalore to Mumbai: {dist:.2f} km")
    
    # Test hospital search
    print(f"\nFinding hospitals near Bangalore...")
    hospitals = find_nearby_hospitals(bangalore_lat, bangalore_lon, radius_km=10)
    for h in hospitals[:3]:
        print(f"  {h['name']}: {h['distance_km']} km away")
