from casadi import numpy as np
from casadi import *
import casadi

import PrintUtil
from ArmDynamics import Arm, Prototype
import ArmKinematics
from Integration import rk4, euler

def DivorcedArm(Arm: Arm, qInit: np.ndarray, qFinal: np.ndarray, N: int) -> casadi.Opti:
    solver = casadi.Opti()
    solver.solver('ipopt')

    u1_max = Arm.proximal.maxtorque
    u2_max = Arm.distal.maxtorque
    
    J1_min = Arm.proximal.minangle
    J1_max = Arm.proximal.maxangle
    J2_min = Arm.distal.minangle
    J2_max = Arm.distal.maxangle

    J1_init = qInit[0, 0]
    J2_init = qInit[1, 0]
    J1dot_init = qInit[2, 0]
    J2dot_init = qInit[3, 0]
    J1_target = qFinal[0, 0]
    J2_target = qFinal[1, 0]
    J1dot_target = qFinal[2, 0]
    J2dot_target = qFinal[3, 0]

    X = solver.variable(4, N + 1) #Arm State
    # for k in range(N + 1):
    #     solver.set_initial(X[0, k], J1_init + k / (N+1) * (J1_target - J1_init))
    #     solver.set_initial(X[1, k], J2_init + k / (N+1) * (J2_target - J2_init))
    U = solver.variable(2, N) #Torques
    T = solver.variable(1) #Total Time

    #Initial conditions
    solver.subject_to(X[0, 0] == J1_init)
    solver.subject_to(X[1, 0] == J2_init)
    solver.subject_to(X[2, 0] == J1dot_init)
    solver.subject_to(X[3, 0] == J2dot_init)

    #Final conditions
    solver.subject_to(X[0, N] == J1_target)
    solver.subject_to(X[1, N] == J2_target)
    solver.subject_to(X[2, N] == J1dot_target)
    solver.subject_to(X[3, N] == J2dot_target)

    #Positional Constrains
    solver.subject_to(solver.bounded(J1_min, X[0, :], J1_max))
    solver.subject_to(solver.bounded(J2_min, X[1, :], J2_max))
    
    #for k in range(N):
    #    solver.subject_to(CartesianBoundsViolation(X[:, k]))

    #Control Constraints
    solver.subject_to(solver.bounded(-u1_max, U[0, :], u1_max))
    solver.subject_to(solver.bounded(-u2_max, U[1, :], u2_max))

    #Time is not negative
    solver.subject_to(T > 0)
    dT = T/N
    for k in range(N):
        solver.subject_to(X[:, k+1] == euler(Arm.ArmDynamics, X[:, k], U[:, k], dT)) #Dynamics
    #Minimize time
    solver.minimize(T)

    return solver, X, U, T


if __name__ == "__main__":
    N = 15
    PrototypeArm = Prototype()
    print(PrototypeArm.proximal.minangle, PrototypeArm.proximal.maxangle)
    print(PrototypeArm.distal.minangle, PrototypeArm.distal.maxangle)
    qInit = np.zeros((4, 1))
    qInit[0, 0] = 0.5
    qInit[1, 0] = 0
    qFinal = np.zeros((4, 1))
    qFinal[0, 0] = 2
    qFinal[1, 0] = -3
    
    solver, X, U, T = DivorcedArm(PrototypeArm, qInit, qFinal, N)
    solution = solver.solve()
    resultant_states = solution.value(X)
    dT = solution.value(T)/N

    for i in range(N + 1):
        q = resultant_states[0:2, i]
        PrintUtil.printArm(q, PrototypeArm.proximal.length, PrototypeArm.proximal.length, i/(N+1))
        # j1x = sin(q[0]) * PrototypeArm.proximal.length
        # j1y = cos(q[1]) * PrototypeArm.proximal.length
        # print("(", j1x, ", ", j1y, ")")
        # x = ArmKinematics.forwardKinematics(q, PrototypeArm.proximal.length, PrototypeArm.distal.length)
        # print("(", x[0, 0], ", ", x[1, 0], ")")
        #print("(", i * dT, ", ", q[0], ")")
        #print("(", i * dT, ", ", q[1], ")")
    #print(resultant_states)