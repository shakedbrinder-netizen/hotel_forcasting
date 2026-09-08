import math
import pandas as pd
from typing import Dict


# High-Efficiency Ratios (Optimal_25p) extracted from historical EDA.
# Reading: "For 100 occupied rooms, we need X employees"
EFFICIENCY_TARGETS = {
    'DT': {
        'Housekeeping': {'Monday': 15.22, 'Tuesday': 14.74, 'Wednesday': 15.16, 'Thursday': 13.53, 'Friday': 12.40, 'Saturday': 16.35, 'Sunday': 16.50},
        'Kitchen':      {'Monday': 18.00, 'Tuesday': 18.00, 'Wednesday': 20.00, 'Thursday': 16.00, 'Friday': 15.00, 'Saturday': 18.00, 'Sunday': 16.00},
        'Waiters':      {'Monday': 10.00, 'Tuesday': 10.00, 'Wednesday': 10.00, 'Thursday': 8.00,  'Friday': 15.00, 'Saturday': 12.00, 'Sunday': 10.00} 
    },
    'DJ': {
        'Housekeeping': {'Monday': 12.28, 'Tuesday': 11.52, 'Wednesday': 11.34, 'Thursday': 11.65, 'Friday': 9.97,  'Saturday': 11.56, 'Sunday': 12.59},
        'Kitchen':      {'Monday': 10.00, 'Tuesday': 10.00, 'Wednesday': 10.00, 'Thursday': 9.00,  'Friday': 8.00,  'Saturday': 9.00,  'Sunday': 10.00},
        'Waiters':      {'Monday': 10.00, 'Tuesday': 10.00, 'Wednesday': 10.00, 'Thursday': 9.00,  'Friday': 18.00, 'Saturday': 14.00, 'Sunday': 10.00} 
    }
}

# Hard Constraints 
# Minimum staff required by hotel policy, even if occupancy is near zero.
MINIMUM_STAFF = {
    'DT': {'Housekeeping': 10, 'Kitchen': 5, 'Waiters': 3},
    'DJ': {'Housekeeping': 8,  'Kitchen': 4, 'Waiters': 3}
}

# --- 2. OPTIMIZATION ENGINE ---

def optimize_staffing(hotel_id: str, target_date: str, predicted_upper_bound_rooms: int) -> Dict[str, int]:
    """
    Calculates the optimal number of employees needed for a specific date.
    
    How it works (The Optimization Logic):
    1. Safety First (Buffer): We use the 'predicted_upper_bound_rooms' (the maximum 
       expected rooms from our AI confidence interval) instead of the average forecast. 
       This prevents understaffing if a sudden occupancy spike occurs.
    2. High-Efficiency Target: We calculate the required staff by applying the 25th 
       percentile ratio (Optimal_25p) extracted from EDA. This ratio represents the 
       hotel's historical "best performing days" for that specific day of the week, 
       safely cutting out historical overstaffing waste.
    3. Hard Constraints: We compare this calculated number against the 'MINIMUM_STAFF' 
       rule (the absolute minimum employees required by hotel policy).
    4. Final Decision: The function returns the highest number between the Efficiency 
       Target and the Hard Constraint to ensure quality and legal compliance are never compromised.
    """
    day_name = pd.to_datetime(target_date).day_name()
    recommendations = {}
    
    for dept in ['Housekeeping', 'Kitchen', 'Waiters']:
        # 1. Quality & Efficiency Constraint
        ratio_optimal = EFFICIENCY_TARGETS[hotel_id][dept][day_name]
        employees_for_quality = (predicted_upper_bound_rooms / 100.0) * ratio_optimal
        
        # 2. Hard Minimum Constraint
        minimum_required = MINIMUM_STAFF[hotel_id][dept]
        
        # 3. Resolution (Max between the two constraints, rounded up to ensure full shifts)
        optimal_employees = math.ceil(max(employees_for_quality, minimum_required))
        
        recommendations[dept] = optimal_employees
        
    return recommendations