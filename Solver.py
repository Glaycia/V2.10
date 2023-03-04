from casadi import numpy as np
from casadi import *
import casadi

from Spline import *
import PrintUtil
from ArmDynamics import Arm, Prototype, Hogfish
import ArmKinematics
from Integration import rk4, euler
from Constraints import *

def DivorcedArm(Arm: Arm, qInit: np.ndarray, qFinal: np.ndarray, N: int, constraints, simplified: bool, use_rk4: bool, text_output: bool = True) -> casadi.Opti:
    solver = casadi.Opti()
    solver.solver('ipopt')

    if not text_output:
        p_opts = dict(print_time=True, verbose=False)
        s_opts = dict(print_level=0)
        solver.solver("ipopt", p_opts, s_opts)
    
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

    for k in range(N + 1):
        x = cos(X[0, k]) * Arm.proximal.length + cos(X[1, k]) * Arm.distal.length
        y = sin(X[0, k]) * Arm.proximal.length + sin(X[1, k]) * Arm.distal.length
        
        for constraint in constraints:
            SmoothConstraintRectangle(solver, x, y, constraint)

    for k in range(N):
        x0 = cos(X[0, k]) * Arm.proximal.length + cos(X[1, k]) * Arm.distal.length
        y0 = sin(X[0, k]) * Arm.proximal.length + sin(X[1, k]) * Arm.distal.length
        x1 = cos(X[0, k + 1]) * Arm.proximal.length + cos(X[1, k + 1]) * Arm.distal.length
        y1 = sin(X[0, k + 1]) * Arm.proximal.length + sin(X[1, k + 1]) * Arm.distal.length
        
        for constraint in constraints:
            SmoothConstraintRectangle(solver, (x0+x1)/2, (y0+y1)/2, constraint)

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

def InitialGuessFRC(Arm: Arm, qInit: np.ndarray, qFinal: np.ndarray, N: int, body_height: float):
    x_init = ArmKinematics.forwardKinematics(qInit[0:2, 0], Arm.proximal.length, Arm.distal.length)[0:2, 0].T[:]
    x_final = ArmKinematics.forwardKinematics(qFinal[0:2, 0], Arm.proximal.length, Arm.distal.length)[0:2, 0].T[:] #Convert it to a vector

    waypoints = [
        Waypoint(p=x_init,v=np.array([0, 0]), a=np.array([0, 0])),
        Waypoint(p=x_final,v=np.array([0, 0]), a=np.array([0, 0]))
    ]

    if x_init[0] * x_final[0] < 0: #If on opposite sides,
        SwingthruSplineVel = 0.3
        waypoints.insert(1, Waypoint(p=np.array([0, Arm.proximal.length - Arm.distal.length]),v=np.array([sign(x_final[0]) * SwingthruSplineVel, 0]), a=np.array([0, 0])))

    if x_init[1] < body_height:
        HeightVelocity = 0.5
        waypoints.insert(1, Waypoint(p=np.array([x_init[0], body_height]),v=np.array([0, HeightVelocity]), a=np.array([0, 0])))

    if x_final[1] < body_height:
        HeightVelocity = 0.5
        waypoints.insert(len(waypoints) - 1, Waypoint(p=np.array([x_final[0], body_height]),v=np.array([0, -HeightVelocity]), a=np.array([0, 0])))
    
    path = Spline(waypoints, N + 1)
    init_guess_points = path.points()

    # for i in range(N + 1):
    #     print("(", init_guess_points[i][0], ",", init_guess_points[i][1], ")")

    parsable_states = np.zeros((4, N + 1))
    for i in range(len(init_guess_points)):
        qpos = ArmKinematics.inverseKinematics(init_guess_points[i], Arm.proximal.length, Arm.distal.length)
        parsable_states[0, i] = qpos[0]
        parsable_states[1, i] = qpos[1]
    
    # print(parsable_states)
    return parsable_states

def Bisolve(Arm: Arm, qInit: np.ndarray, qFinal: np.ndarray, N: int, constraints, text_output: bool = True) -> casadi.Opti:
    solver_1, X_1, U_1, T_1 = DivorcedArm(Arm, qInit, qFinal, N, constraints, True, False, text_output)
    solver_2, X_2, U_2, T_2 = DivorcedArm(Arm, qInit, qFinal, N, constraints, False, True, text_output)

    x_init = ArmKinematics.forwardKinematics(qInit[0:2, 0], Arm.proximal.length, Arm.distal.length)[0:2, 0].T[:]
    x_final = ArmKinematics.forwardKinematics(qFinal[0:2, 0], Arm.proximal.length, Arm.distal.length)[0:2, 0].T[:] #Convert it to a vector
    if x_init[1] < -0.08255 and x_final[1] < -0.08255:
        solver_1.set_initial(X_1, InitialGuessFRC(Arm, qInit, qFinal, N, -0.08255))
    solution_1 = solver_1.solve()

    solver_2.set_initial(X_2, solution_1.value(X_1))
    solver_2.set_initial(U_2, solution_1.value(U_1))
    solver_2.set_initial(T_2, solution_1.value(T_1))

    solution_2 = solver_2.solve()

    # SolutionDesmos(solution_1, X_1, T_1)
    # SolutionDesmos(Arm, solution_2, X_2, T_2)

    return solution_2, X_2, U_2, T_2

def Trisolve(Arm: Arm, qInit: np.ndarray, qFinal: np.ndarray, N: int, constraints, text_output: bool = True) -> casadi.Opti:
    solver_1, X_1, U_1, T_1 = DivorcedArm(Arm, qInit, qFinal, N, constraints, True, False, text_output)
    solver_2, X_2, U_2, T_2 = DivorcedArm(Arm, qInit, qFinal, N, constraints, False, False, text_output)
    solver_3, X_3, U_3, T_3 = DivorcedArm(Arm, qInit, qFinal, N, constraints, False, True, text_output)

    solution_1 = solver_1.solve()

    solver_2.set_initial(X_2, solution_1.value(X_1))
    solver_2.set_initial(U_2, solution_1.value(U_1))
    solver_2.set_initial(T_2, solution_1.value(T_1))

    solution_2 = solver_2.solve()

    solver_3.set_initial(X_3, solution_2.value(X_2))
    solver_3.set_initial(U_3, solution_2.value(U_2))
    solver_3.set_initial(T_3, solution_2.value(T_2))

    solution_3 = solver_3.solve()

    # SolutionDesmos(solution_1, X_1, T_1)
    # SolutionDesmos(solution_2, X_2, T_2)
    # SolutionDesmos(Arm, solution_3, X_3, T_3)

    return solution_3, X_3, U_3, T_3

def SolutionDesmos(Arm: Arm, solution, X, T, T_offset = 0):
    resultant_states = solution.value(X)
    N = X.size2() - 1
    dT = solution.value(T)/N

    for i in range(N + 1):
        q = resultant_states[0:2, i]
        PrintUtil.printArm(q, Arm.proximal.length, Arm.distal.length, i * dT + T_offset)



if __name__ == "__main__":
    N = 50
    PrototypeArm = Hogfish()
    #print(PrototypeArm.proximal.minangle, PrototypeArm.proximal.maxangle)
    #print(PrototypeArm.distal.minangle, PrototypeArm.distal.maxangle)
    x = np.zeros((2, 1))
    x[0, 0] = 0.6
    x[1, 0] = -0.2
    qIK = ArmKinematics.inverseKinematics(x, PrototypeArm.proximal.length, PrototypeArm.distal.length)
    x[0, 0] = -0.6
    x[1, 0] = -0.2
    qIK2 = ArmKinematics.inverseKinematics(x, PrototypeArm.proximal.length, PrototypeArm.distal.length)
    xDot = np.zeros((2, 1))
    xDot[0, 0] = 0
    xDot[1, 0] = 0
    qDotIK2 = ArmKinematics.inverseVelocities(qIK2, xDot, PrototypeArm.proximal.length, PrototypeArm.distal.length)
    print(qIK2[0])
    print(qIK2[1])

    qInit = np.zeros((4, 1))
    qInit[0, 0] = qIK[0]
    qInit[1, 0] = qIK[1]
    qFinal = np.zeros((4, 1))
    qFinal[0, 0] = qIK2[0]
    qFinal[1, 0] = qIK2[1]
    
    qFinal[2, 0] = qDotIK2[0]
    qFinal[3, 0] = qDotIK2[1]
    
    clearance = 0.127
    rule_constraint = ConstraintParameter(x0= 1.7018-clearance, y0= -0.17145 -0.20955 + clearance, x1= -1.7018+clearance, y1=1.9812, constrained_within=True)
    robot_body = ConstraintParameter(x0= 0.8382/2 + clearance, y0= -0.20955 + clearance, x1=-0.8382/2 - clearance, y1=-1, constrained_within=False)

    solution, X, U, T = Bisolve(PrototypeArm, qInit, qFinal, N, [rule_constraint, robot_body])

    resultant_states = solution.value(X)
    resultant_controls = solution.value(U)
    dT = solution.value(T)/N
    
    SolutionDesmos(Arm, solution, X, T)
    
    #print(resultant_states.shape)