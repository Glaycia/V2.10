from ArmDynamics import Arm, Hogfish
from Constraints import *
import ArmKinematics
import numpy as np
from Solver import Bisolve, SolutionDesmos

import TrajectoryWriter

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

            if not check_valid_state(Arm, x_pos):
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
        solution, X, U, T = Bisolve(Arm, waypoints[i].as2D(), waypoints[i + 1].as2D(), Nper, constraints)

        SolutionDesmos(Arm, solution, X, T, T_offset= list_of_time[len(list_of_time) - 1])

        state_array = solution.value(X)
        control_array = solution.value(U)
        segment_duration = solution.value(T)
        for j in range(Nper):
            list_of_states.append(state_array[0:4, j])
            list_of_controls.append(control_array[0:2, j])
            list_of_time.append(list_of_time[len(list_of_time) - 1] + segment_duration/Nper)
        if i == len(waypoints) - 2:
            list_of_states.append(waypoints[i + 1].as2D())

    return list_of_states, list_of_controls, list_of_time


if __name__ == "__main__":
    Arm = Hogfish()
    p0 = ArmState(Arm, np.array([np.pi/2, -np.pi/2, 0, 0]), is_jointspace=True)
    p1 = ArmState(Arm, np.array([0.2, 1, 0, 2]), is_jointspace=False)
    p2 = ArmState(Arm, np.array([1, 1, 0, 0]), is_jointspace=False)

    clearance = 0.127
    rule_constraint = ConstraintParameter(x0= 1.7018-clearance, y0= -0.17145 -0.20955 + clearance, x1= -1.7018+clearance, y1=1.9812, constrained_within=True)
    robot_body = ConstraintParameter(x0= 0.8382/2 + clearance, y0= -0.20955 + clearance, x1=-0.8382/2 - clearance, y1=-1, constrained_within=False)

    states, controls, times = solve(Arm, [p0, p1, p2], 30, [rule_constraint, robot_body])

    TrajectoryWriter.write_trajectory("Default_To_Shelf", p0.q, p2.q, states, controls, times)