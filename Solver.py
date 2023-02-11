from casadi import numpy as np
from casadi import *
import casadi

import PrintUtil
from ArmDynamics import Arm, Prototype
import ArmKinematics
from Integration import rk4, euler

def DivorcedArm(Arm: Arm, qInit: np.ndarray, qFinal: np.ndarray, N: int, simplified: bool = False, use_rk4: bool = False) -> casadi.Opti:
    solver = casadi.Opti()
    solver.solver('ipopt')
    
    u1_max = Arm.proximal.max_torque
    u2_max = Arm.distal.max_torque
    
    J1_min = Arm.proximal.min_angle
    J1_max = Arm.proximal.max_angle
    J2_min = Arm.distal.min_angle
    J2_max = Arm.distal.max_angle

    J1_init = qInit[0, 0]
    J2_init = qInit[1, 0]
    J1dot_init = qInit[2, 0]
    J2dot_init = qInit[3, 0]
    J1_target = qFinal[0, 0]
    J2_target = qFinal[1, 0]
    J1dot_target = qFinal[2, 0]
    J2dot_target = qFinal[3, 0]

    X = solver.variable(4, N + 1) #Arm State
    # if simplified == True:
    #     for k in range(N + 1):
    #         solver.set_initial(X[0, k], J1_init + k / (N+1) * (J1_target - J1_init))
    #         solver.set_initial(X[1, k], J2_init + k / (N+1) * (J2_target - J2_init))
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

    for k in range(N + 1): #Do you really need N + 1, N probably just suffices
        solver.subject_to(sin(X[0, k]) * Arm.proximal.length + sin(X[1, k]) * Arm.distal.length > 0)
    #for k in range(N):
    #    solver.subject_to(CartesianBoundsViolation(X[:, k]))

    #Control Constraints
    solver.subject_to(solver.bounded(-u1_max, U[0, :], u1_max))
    solver.subject_to(solver.bounded(-u2_max, U[1, :], u2_max))

    #Time is not negative
    solver.subject_to(T > 0)
    dT = T/N
    for k in range(N):
        if simplified:
            solver.subject_to(X[:, k+1] == euler(Arm.StabilizedDynamics, X[:, k], U[:, k], dT)) #Dynamics
        elif use_rk4:
            solver.subject_to(X[:, k+1] == euler(Arm.ArmDynamics, X[:, k], U[:, k], dT)) #Dynamics
        else:
            solver.subject_to(X[:, k+1] == rk4(Arm.ArmDynamics, X[:, k], U[:, k], dT)) #Dynamics
    #Minimize time
    solver.minimize(T)

    return solver, X, U, T

def Multisolve(Arm: Arm, qInit: np.ndarray, qFinal: np.ndarray, N: int) -> casadi.Opti:
    solver_1, X_1, U_1, T_1 = DivorcedArm(Arm, qInit, qFinal, N, True, False)
    solver_2, X_2, U_2, T_2 = DivorcedArm(Arm, qInit, qFinal, N, False, False)
    solver_3, X_3, U_3, T_3 = DivorcedArm(Arm, qInit, qFinal, N, False, True)

    solution_1 = solver_1.solve()

    solver_2.set_initial(X_2, solution_1.value(X_1))
    solver_2.set_initial(U_2, solution_1.value(U_1))
    solver_2.set_initial(T_2, solution_1.value(T_1))

    solution_2 = solver_2.solve()

    solver_3.set_initial(X_3, solution_2.value(X_2))
    solver_3.set_initial(U_3, solution_2.value(U_2))
    solver_3.set_initial(T_3, solution_2.value(T_2))

    solution_3 = solver_3.solve()
    return solution_3, X_3, U_3, T_3


if __name__ == "__main__":
    N = 25
    PrototypeArm = Prototype()
    #print(PrototypeArm.proximal.minangle, PrototypeArm.proximal.maxangle)
    #print(PrototypeArm.distal.minangle, PrototypeArm.distal.maxangle)
    qInit = np.zeros((4, 1))
    qInit[0, 0] = 0.5
    qInit[1, 0] = -0.3
    qFinal = np.zeros((4, 1))
    qFinal[0, 0] = 2
    qFinal[1, 0] = -pi
    
    solution, X, U, T = Multisolve(PrototypeArm, qInit, qFinal, N)

    resultant_states = solution.value(X)
    dT = solution.value(T)/N

    for i in range(N + 1):
        q = resultant_states[0:2, i]
        PrintUtil.printArm(q, PrototypeArm.proximal.length, PrototypeArm.proximal.length, i * dT)
        # j1x = sin(q[0]) * PrototypeArm.proximal.length
        # j1y = cos(q[1]) * PrototypeArm.proximal.length
        # print("(", j1x, ", ", j1y, ")")
        # x = ArmKinematics.forwardKinematics(q, PrototypeArm.proximal.length, PrototypeArm.distal.length)
        # print("(", x[0, 0], ", ", x[1, 0], ")")
        # print("(", i * dT, ", ", q[0], ")")
        # print("(", i * dT, ", ", q[1], ")")
    #print(resultant_states)