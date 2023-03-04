from ArmDynamics import Arm, Hogfish, Prototype
from Constraints import *
import ArmKinematics, Constraints
import numpy as np
from Solver import Bisolve, SolutionDesmos

import TrajectoryWriter, JavaWriter

def check_valid_state(Arm, X):
    x = X[0]
    y = X[1]
    return not (np.abs(x) + np.abs(y) > 0 and Arm.proximal.length != Arm.distal.length)
class ArmState:
    #X given in [xpos, ypos, xvel, yvel], a 1-D vector
    def __init__(self, Arm: Arm, input, is_jointspace = False):
        if is_jointspace:
            self.q = input
        else:
            x = input

            self.x_init = x

            x_pos = x[0:2]
            x_vel = x[2:4]

            if check_valid_state(Arm, x_pos):
                print("Outside of arm bounds, regularizing radially")

            q_pos = ArmKinematics.inverseKinematics(x_pos, Arm.proximal.length, Arm.distal.length)
            q_vel = ArmKinematics.inverseVelocities(q_pos, x_vel, Arm.proximal.length, Arm.distal.length)


            self.q = np.empty((4))
            self.q[0:2] = q_pos
            self.q[2:4] = q_vel

    def as2D(self):
        q_2D = np.array([self.q]).T
        return q_2D

def solve(Arm: Arm, waypoints: ArmState, Nper: int, constraints):
    list_of_states = []
    list_of_controls = []
    list_of_time = [0]
    for i in range(len(waypoints) - 1):
        solution, X, U, T = Bisolve(Arm, waypoints[i].as2D(), waypoints[i + 1].as2D(), Nper, constraints, text_output= False)

        SolutionDesmos(Arm, solution, X, T, T_offset= list_of_time[len(list_of_time) - 1])

        state_array = solution.value(X)
        control_array = solution.value(U)
        segment_duration = solution.value(T)
        for j in range(Nper):
            list_of_states.append(state_array[0:4, j])
            list_of_controls.append(control_array[0:2, j])
            list_of_time.append(list_of_time[len(list_of_time) - 1] + segment_duration/Nper)
        if i == len(waypoints) - 2:
            list_of_states.append(waypoints[i + 1].q)

    return list_of_states, list_of_controls, list_of_time


def scoring_waypoints():
    p0 = ArmState(Arm, np.array([np.pi/2, -np.pi/2, 0, 0]), is_jointspace=True)
    p1 = ArmState(Arm, np.array([1.3, 1.1, 1, 0]), is_jointspace=False)
    p2 = ArmState(Arm, np.array([(40 + (3 + 13)) * inches_to_meters, 46 * inches_to_meters - howfararmjointofffloor, -0.5, -1]), is_jointspace=False)
    p3 = ArmState(Arm, np.array([np.pi/2, -np.pi/2, 0, 0]), is_jointspace=True)
    return [p0, p1, p2, p3]
def scoring_constraint():
    return ConstraintParameter(x0 = (16 + (3 + 13) + 5) * inches_to_meters, y0 = -1, x1 = (70 + (3 + 13) + 5) * inches_to_meters, y1 = (36+5) * inches_to_meters - howfararmjointofffloor, constrained_within= False)
if __name__ == "__main__":
    Arm = Hogfish()

    inches_to_meters = 0.0254

    howfararmjointofffloor = 8.25 * inches_to_meters

    p0 = ArmState(Arm, np.array([np.pi/2, -np.pi/2, 0, 0]), is_jointspace=True)
    p1 = ArmState(Arm, np.array([1, 0.5, 4.5, 4.5]), is_jointspace=False)
    p2 = ArmState(Arm, np.array([-1.5, -0.2, 0, 0]), is_jointspace=False)

    Hogfish_Constraints = Constraints.Hogfish_Constraints()
    Prototype_Constraints = Constraints.Prototype_Constraints()

    Hogfish_ScoringConstraint = scoring_constraint()
    
    #Hogfish_Constraints.append(Hogfish_ScoringConstraint)

    states, controls, times = solve(Arm, [p0, p1, p0], 50, Hogfish_Constraints)

    # for i in range(len(times)):
    #     print("(", times[i] + 6, ",", controls[i][0]/100, ")")
    #     print("(", times[i] + 6, ",", controls[i][1]/100, ")")
    #     print("m")
    #     print("m")
    #     print("m")

    #JavaWriter.write_trajectory("ScoreMotion", p0.q, p3.q, states, controls, times)
    #TrajectoryWriter.write_trajectory("PrototypeSwingup", p0.q, p2.q, states, controls, times)