import numpy as np

class FeatureExtractor:
    def __init__(self, metadata):
        self.feature_cols = metadata.get('feature_cols', [])
        
    def extract(self, angle_window, risk_window, rula_l_window, rula_r_window, reba_l_window, reba_r_window, current_time):
        """
        Produce the feature dict matching self.feature_cols using the 60-element deques.
        """
        features = {}
        
        # Helper to safely extract series
        def get_series_from_dicts(window, key):
            # window contains dicts
            return [frame.get(key, 0.0) for frame in window]

        # Map base columns to series
        series_map = {}
        
        # Angle series
        # We assume angle_window contains dicts of joint angles
        if len(angle_window) > 0:
            for k in angle_window[0].keys():
                series_map[k] = get_series_from_dicts(angle_window, k)
                
        # Risk series
        series_map['global_risk_score'] = list(risk_window)
        series_map['RULA_L_Final'] = list(rula_l_window)
        series_map['RULA_R_Final'] = list(rula_r_window)
        series_map['REBA_L_Final'] = list(reba_l_window)
        series_map['REBA_R_Final'] = list(reba_r_window)
        
        # Need to parse other RULA/REBA subscores but we might not have historical tracking for them all.
        # But wait! The metadata requires things like RULA_R_Upper_Arm_Score_mean...
        # If we don't have them in the window, we'll extract them as 0 to avoid breaking.
        
        for col in self.feature_cols:
            if col in ['hour_of_day', 'day_of_week']:
                import datetime
                dt = datetime.datetime.fromtimestamp(current_time)
                features['hour_of_day'] = dt.hour
                features['day_of_week'] = dt.weekday()
                continue
            
            # Simple global risk
            if col == 'global_risk_score':
                features[col] = series_map['global_risk_score'][-1] if series_map['global_risk_score'] else 0.0
                continue
                
            # Parse stat
            for stat in ['_mean', '_std', '_max', '_p95']:
                if col.endswith(stat):
                    base = col[:-len(stat)]
                    s = series_map.get(base, [0.0])
                    if not s: s = [0.0]
                    if stat == '_mean': features[col] = np.mean(s)
                    elif stat == '_std': features[col] = np.std(s)
                    elif stat == '_max': features[col] = np.max(s)
                    elif stat == '_p95': features[col] = np.percentile(s, 95) if len(s) > 0 else 0.0
                    break
                    
            # Parse lag
            for lag in ['_lag1', '_lag3', '_lag6', '_lag12']:
                if col.endswith(lag):
                    base = col[:-len(lag)]
                    if base.endswith('_mean'): base = base[:-5] # e.g. Neck_Flexion_deg_mean_lag1
                    s = series_map.get(base, [0.0])
                    lag_val = int(lag[4:])
                    if len(s) >= lag_val + 1:
                        features[col] = s[-(lag_val + 1)]
                    else:
                        features[col] = s[0] if s else 0.0
                    break
                    
            # Parse rolling
            for roll in ['_roll6', '_roll12', '_roll24']:
                if roll in col:
                    parts = col.split(roll)
                    base = parts[0]
                    if base.endswith('_mean'): base = base[:-5]
                    stat2 = parts[1] # _mean or _std
                    s = series_map.get(base, [0.0])
                    roll_val = int(roll[5:])
                    subset = s[-roll_val:] if len(s) >= roll_val else s
                    if not subset: subset = [0.0]
                    if stat2 == '_mean': features[col] = np.mean(subset)
                    elif stat2 == '_std': features[col] = np.std(subset)
                    break
                    
        # Ensure all required features are set (default to 0.0)
        for req_col in self.feature_cols:
            if req_col not in features:
                features[req_col] = 0.0
                
        # Fill Anomaly Score input too if needed as feature
        if 'anomaly_score' in features and features['anomaly_score'] == 0:
            # We predict anomaly score inside ai_engine using if_features
            features['anomaly_score'] = 0.0 
                
        return features
