import pandas as pd
import numpy as np

def preprocess_kill_data(kill_data):
    # Add 'kill_distance' column
    kill_data['kill_distance'] = np.sqrt(
        (kill_data['killer_position_x'] - kill_data['victim_position_x']) ** 2 +
        (kill_data['killer_position_y'] - kill_data['victim_position_y']) ** 2
    )
    # Filter invalid positions
    positions = ['killer_position_x', 'killer_position_y', 'victim_position_x', 'victim_position_y']
    for pos in positions:
        kill_data = kill_data[(kill_data[pos] >= 0) & (kill_data[pos] <= 800000)]

    return kill_data
