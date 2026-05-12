from dataclasses import dataclass

@dataclass
class DartboardScorer:
    dartboard_numbers = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    
    double_bull_radius = 90 
    bull_radius = 159
    treble_inner_radius = 990
    treble_outer_radius = 1070
    double_inner_radius = 1620
    double_outer_radius = 1700 

    segment_angle = 18

    calibration_angles = [
        351,  # Between 20 and 5
        99,   # Between 6 and 10
        171,  # Between 3 and 17
        279   # Between 11 and 14
    ]