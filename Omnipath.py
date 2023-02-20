from ArmDynamics import Arm, Prototype, Hogfish
import ArmKinematics, Constraints
from Solver import *

def get_possible_states(Arm: Arm, constraints):
    possible_states = []
    scan_up = False
    stepsize = 0.5
    for a1 in range(Arm.proximal.min_angle, Arm.proximal.max_angle, stepsize):
        if scan_up:
            scan_up = False
            for a2 in reversed(range(Arm.distal.min_angle, Arm.proximal.max_angle, stepsize)):
                pass
                
        else:
            scan_up = True
            for a2 in range(Arm.distal.min_angle, Arm.proximal.max_angle, stepsize):
                pass