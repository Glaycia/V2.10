import json, os
from datetime import datetime
from json import JSONEncoder
import numpy as np

def get_path_to_script():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    return path_to_script
def time_as_string() -> str:
    currentDateAndTime = datetime.now()
    return currentDateAndTime.strftime("%m_%d_%H-%M-%S")
def write_trajectory(name, initial_state, final_state, state_data, control_data, time_data):
    data = {
        "name": name,
        "initial_state": initial_state,
        "final_state": final_state,
        "state_data": state_data,
        "control_data": control_data,
        "time_data": time_data
    }
    with open(get_path_to_script() + "/TrajectoryData/" + name + time_as_string() + ".json", "w") as outfile:
        json.dump(data, outfile, indent=4, cls=NumpyArrayEncoder)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

if __name__ == "__main__":
    name = "trajectoryL9"
    initial_state = np.array([0, 0])
    final_state = np.array([1, 1])
    trajectory_data = [
        np.array([0.5, 0.5]),
        np.array([0.75, 0.75])
        ]
    time_data = [
        0.5,
        0.76
    ]
    