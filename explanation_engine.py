"""
Risk Explanation Engine using SHAP

Provides interpretable explanations for accident risk predictions:
- SHAP value calculations for feature importance
- Natural language risk summaries
- Visual explanation generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")


class RiskExplainer:
    """Explain accident risk predictions using SHAP values."""
    
    def __init__(self, model_pipeline):
        """
        Initialize explainer with trained model.
        
        Args:
            model_pipeline: Trained sklearn pipeline with preprocessor and model
        """
        self.pipeline = model_pipeline
        self.explainer = None
        self.feature_names = None
        
        if SHAP_AVAILABLE:
            try:
                # Get the final model from pipeline
                if hasattr(model_pipeline, 'named_steps'):
                    self.model = model_pipeline.named_steps.get('classifier', model_pipeline)
                else:
                    self.model = model_pipeline
                
                # Initialize SHAP explainer for tree-based models
                self.explainer = shap.TreeExplainer(self.model)
                print("SHAP explainer initialized successfully")
            except Exception as e:
                print(f"SHAP initialization warning: {e}")
    
    def explain_prediction(self, input_data: pd.DataFrame, top_n: int = 5) -> Dict:
        """
        Generate explanation for a single prediction.
        
        Args:
            input_data: DataFrame with single row of features
            top_n: Number of top features to include in explanation
            
        Returns:
            Dictionary with SHAP values, feature importance, and natural language explanation
        """
        try:
            # Make prediction
            prediction_proba = self.pipeline.predict_proba(input_data)[0]
            predicted_class = np.argmax(prediction_proba)
            confidence = prediction_proba[predicted_class]
            
            risk_labels = ["Low", "Medium", "High"]
            predicted_risk = risk_labels[predicted_class]
            
            # If SHAP not available, return basic explanation
            if not SHAP_AVAILABLE or self.explainer is None:
                return self._basic_explanation(input_data, predicted_risk, confidence)
            
            # Transform input through preprocessing pipeline
            try:
                preprocessor = self.pipeline.named_steps.get('preprocessor', None)
                if preprocessor:
                    X_transformed = preprocessor.transform(input_data)
                else:
                    X_transformed = input_data.values
            except:
                X_transformed = input_data.values
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_transformed)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values_for_class = shap_values[predicted_class][0]
            else:
                shap_values_for_class = shap_values[0]
            
            # Get feature importance
            feature_importance = np.abs(shap_values_for_class)
            
            # Get top features
            top_indices = np.argsort(feature_importance)[-top_n:][::-1]
            
            # Extract feature names (approximation since we have transformed features)
            original_features = list(input_data.columns)
            
            top_features = []
            for idx in top_indices:
                if idx < len(original_features):
                    feature_name = original_features[idx]
                    impact = shap_values_for_class[idx]
                    importance_pct = (feature_importance[idx] / np.sum(feature_importance)) * 100
                    
                    top_features.append({
                        'feature': feature_name,
                        'value': input_data[feature_name].values[0] if feature_name in input_data.columns else 'N/A',
                        'impact': float(impact),
                        'importance_pct': round(importance_pct, 1),
                        'direction': 'increases' if impact > 0 else 'decreases'
                    })
            
            # Generate natural language explanation
            explanation_text = self._generate_explanation_text(
                predicted_risk, confidence, top_features, input_data
            )
            
            return {
                'predicted_risk': predicted_risk,
                'confidence': round(confidence * 100, 1),
                'top_factors': top_features,
                'explanation': explanation_text,
                'shap_available': True
            }
            
        except Exception as e:
            print(f"SHAP explanation error: {e}")
            return self._basic_explanation(input_data, "Unknown", 0.5)
    
    def _basic_explanation(self, input_data: pd.DataFrame, predicted_risk: str, 
                          confidence: float) -> Dict:
        """Generate basic rule-based explanation when SHAP is unavailable."""
        
        # Extract key features
        factors = []
        
        if 'is_night' in input_data.columns and input_data['is_night'].values[0] == 1:
            factors.append({
                'feature': 'Time of Day',
                'value': 'Night',
                'importance_pct': 25,
                'direction': 'increases'
            })
        
        if 'Weather_conditions' in input_data.columns:
            weather = input_data['Weather_conditions'].values[0]
            if 'Rain' in str(weather):
                factors.append({
                    'feature': 'Weather',
                    'value': weather,
                    'importance_pct': 20,
                    'direction': 'increases'
                })
        
        if 'Speed_limit' in input_data.columns:
            speed = input_data['Speed_limit'].values[0]
            if speed >= 80:
                factors.append({
                    'feature': 'Speed Limit',
                    'value': f'{speed} km/h',
                    'importance_pct': 15,
                    'direction': 'increases'
                })
        
        if 'location_accident_count' in input_data.columns:
            count = input_data['location_accident_count'].values[0]
            if count > 5:
                factors.append({
                    'feature': 'Accident History',
                    'value': f'{count} past accidents',
                    'importance_pct': 30,
                    'direction': 'increases'
                })
        
        explanation_text = f"The prediction is based on {len(factors)} key risk factors."
        
        return {
            'predicted_risk': predicted_risk,
            'confidence': round(confidence * 100, 1),
            'top_factors': factors,
            'explanation': explanation_text,
            'shap_available': False
        }
    
    def _generate_explanation_text(self, predicted_risk: str, confidence: float,
                                   top_features: List[Dict], input_data: pd.DataFrame) -> str:
        """Generate natural language explanation."""
        
        risk_level_text = {
            "Low": "relatively safe conditions",
            "Medium": "moderate risk conditions",
            "High": "high-risk conditions"
        }
        
        intro = f"The model predicts {risk_level_text.get(predicted_risk, 'unknown conditions')} with {confidence*100:.0f}% confidence."
        
        if not top_features:
            return intro + " No specific risk factors identified."
        
        # List top factors
        factor_texts = []
        for i, factor in enumerate(top_features[:3], 1):
            feature_name = factor['feature'].replace('_', ' ').title()
            direction = factor['direction']
            importance = factor['importance_pct']
            
            factor_texts.append(f"{feature_name} ({importance:.0f}% impact, {direction} risk)")
        
        factors_list = ", ".join(factor_texts)
        
        explanation = f"{intro} Key contributing factors: {factors_list}."
        
        # Add actionable insights
        if predicted_risk == "High":
            explanation += " Consider delaying travel, choosing alternative routes, or taking extra precautions."
        elif predicted_risk == "Medium":
            explanation += " Exercise caution and stay alert to changing conditions."
        
        return explanation
    
    def get_feature_importance_summary(self, X_sample: pd.DataFrame, 
                                      num_samples: int = 100) -> Dict:
        """
        Calculate global feature importance across multiple samples.
        
        Args:
            X_sample: Sample dataset for importance calculation
            num_samples: Number of samples to use
            
        Returns:
            Dictionary with global feature importance rankings
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return {'error': 'SHAP not available'}
        
        try:
            # Sample data
            sample = X_sample.sample(min(num_samples, len(X_sample)))
            
            # Transform
            preprocessor = self.pipeline.named_steps.get('preprocessor', None)
            if preprocessor:
                X_transformed = preprocessor.transform(sample)
            else:
                X_transformed = sample.values
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_transformed)
            
            # Average absolute SHAP values
            if isinstance(shap_values, list):
                importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                importance = np.abs(shap_values).mean(axis=0)
            
            # Map to feature names (simplified)
            feature_names = list(sample.columns)
            
            importance_dict = {}
            for i, name in enumerate(feature_names):
                if i < len(importance):
                    importance_dict[name] = float(importance[i])
            
            # Sort by importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'feature_importance': dict(sorted_features[:10]),
                'num_samples': len(sample)
            }
            
        except Exception as e:
            print(f"Global importance calculation error: {e}")
            return {'error': str(e)}


def format_explanation_for_display(explanation: Dict) -> str:
    """
    Format explanation dictionary into readable text for UI display.
    
    Args:
        explanation: Dictionary from explain_prediction()
        
    Returns:
        Formatted string for display
    """
    lines = []
    lines.append(f"**Predicted Risk**: {explanation['predicted_risk']} ({explanation['confidence']}% confidence)")
    lines.append("")
    lines.append("**Top Contributing Factors**:")
    
    for i, factor in enumerate(explanation.get('top_factors', []), 1):
        feature = factor.get('feature', 'Unknown')
        importance = factor.get('importance_pct', 0)
        direction = factor.get('direction', 'affects')
        value = factor.get('value', 'N/A')
        
        lines.append(f"{i}. **{feature}**: {value} ({importance}% impact - {direction} risk)")
    
    lines.append("")
    lines.append(f"**Explanation**: {explanation.get('explanation', 'No explanation available.')}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("Risk Explainer module loaded")
    print(f"SHAP available: {SHAP_AVAILABLE}")
