import json, os
from datetime import datetime
from json import JSONEncoder
import numpy as np

def get_path_to_script():
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    return path_to_script
def time_as_string() -> str:
    currentDateAndTime = datetime.now()
    return currentDateAndTime.strftime("%m-%d, %H:%M:%S")

def tab_num(n: int):
    return "\t" * n

def java_string(array: np.ndarray) -> str:
    array_string = np.array2string(array, suppress_small=True, separator=", ")
    return array_string.replace("[", "{").replace("]", "}")

def list_of_ndarray_string(input, tabs:int = 0) -> str:
    prefix = tab_num(tabs) + "{\n"
    suffix = tab_num(tabs) + "}"
    content = ""
    for i, array in enumerate(input):
        content += tab_num(tabs + 1) + java_string(array)
        if i != len(input) - 1:
            content += ","
        content += "\n"
    
    return prefix + content + suffix
def write_trajectory(name, initial_state, final_state, state_data, control_data, time_data):
    data = {
        "name": name,
        "initial_state": initial_state,
        "final_state": final_state,
        "state_data": state_data,
        "control_data": control_data,
        "time_data": time_data
    }
    file_path = get_path_to_script() + "/TrajectoryData/" + name + ".java"
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, "w") as outfile:
        creationtime_str = ("//") + "Generated at " + (time_as_string()) + "\n"

        class_str = "public class " + name + " extends ArmPathBase { \n"

        initial_state_str = "\tpublic static final double[] initialState = new double[] " + java_string(initial_state) + ";\n"
        final_state_str = "\tpublic static final double[] finalState = new double[] " + java_string(final_state) + ";\n"
        time_data_str = "\tpublic static final double[] timeData = new double[] \n" + tab_num(3) + str(time_data).replace("[", "{").replace("]", "}").replace(",", ",\n" + tab_num(3)) + ";\n"
        state_data_str = "\tpublic static final double[][] stateData = new double[][] \n " + list_of_ndarray_string(state_data, tabs=2) + ";\n"
        control_data_str = "\tpublic static final double[][] controlData = new double[][] \n " + list_of_ndarray_string(control_data, tabs=2) + ";\n"
        
        end_str = "} //Optimized By Glaciace"

        conglomerate = creationtime_str + class_str + initial_state_str + final_state_str + time_data_str + state_data_str + control_data_str + end_str

        outfile.write(conglomerate)

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

    write_trajectory("Tester", initial_state, final_state, trajectory_data, trajectory_data, time_data)
    