from casadi import numpy as np
import control
from ArmDynamics import Arm, Prototype

def linearizeSystem(Arm, x, u):
    epsilon = 1/(2.0 ** 10)
    states = 4
    inputs = 2
    A = np.empty((states, states))
    B = np.empty((states, inputs))
    #df_i/dx_j
    for i in range(states):
        right = x.copy()
        left = x.copy()
        right[i] = right[i] + epsilon
        left[i] = left[i] - epsilon
        diff = ((Arm.StabilizedDynamics(right, u) - Arm.StabilizedDynamics(left, u))/(2 * epsilon))[:, 0]
        for j in range(states):
            A[j, i] = diff[j]
    #df_i/du_j
    for i in range(inputs):
        right = u.copy()
        left = u.copy()
        right[i] += epsilon
        left[i] -= epsilon
        diff = ((Arm.StabilizedDynamics(x, right) - Arm.StabilizedDynamics(x, left))/(2 * epsilon))[:, 0]
        for j in range(states):
            B[j, i] = diff[j]
    return A, B
 
if __name__ == "__main__":
    r = np.empty((4, 1))
    r[0, 0] = 1
    r[1, 0] = 0
    r[2, 0] = 0
    r[3, 0] = 0
 
    x = np.empty((4, 1))
    x[0, 0] = 1.2
    x[1, 0] = -1
    x[2, 0] = 0.1
    x[3, 0] = 0.05
 
    u = np.empty((2, 1))
    u[0, 0] = 0
    u[1, 0] = 0
 
    deviationTheta = 0.01
    deviationOmega = 0.1
    Q = np.zeros((4, 4))
    Q[0, 0] = 1/deviationTheta**2
    Q[1, 1] = 1/deviationTheta**2
    Q[2, 2] = 1/deviationOmega**2
    Q[3, 3] = 1/deviationOmega**2
 
    controlCost = 10
    R = np.zeros((2, 2))
    R[0, 0] = controlCost
    R[1, 1] = controlCost
 
    A, B = linearizeSystem(Prototype(), x, u)
 
    K, S, E = control.lqr(A, B, Q, R)
 
    #print(K)
    feedback = K @ (r - x)
    #feedforward = StabilizingTorques(x)
    print(feedback)
    #print(feedforward)
 
    print(A)
    print(B)