from casadi import numpy as np
from casadi import *
import casadi
import platform

def ConstraintRectangle(solver: casadi.Opti, x, y, x0: float, y0: float, x1: float, y1: float, constrained_within: bool):
    x_center = (x0 + x1)/2
    y_center = (y0 + y1)/2
    width = fabs(x0 - x1)  #x
    height = fabs(y0 - y1) #y

    #If within, f = 0, gets larger the further away from the constraint rectangle
    f_x = fmax(fabs(x - x_center) - width/2, 0)
    f_y = fmax(fabs(y - y_center) - height/2, 0)

    if constrained_within:
        solver.subject_to(f_x + f_y == 0)
    else:
        solver.subject_to(f_x + f_y > 1e-5)

def SmoothConstraintRectangle(solver: casadi.Opti, x, y, x0: float, y0: float, x1: float, y1: float, constrained_within: bool):
    x_center = (x0 + x1)/2
    y_center = (y0 + y1)/2
    width = fabs(x0 - x1)  #x
    height = fabs(y0 - y1) #y

    if constrained_within:
        f_inside = lambda a, l_b, h_b: -(a - l_b) * (a - h_b)

        f_x = f_inside(x, x_center - width/2, x_center + width/2)
        f_y = f_inside(y, y_center - height/2, y_center + height/2)
        solver.subject_to(f_x + f_y > 0)
    else:
        f = lambda a, q: (sqrt(a**2 + q) + a - sqrt(q))/2
        f_outside = lambda a, q, l_b, h_b: -(f(a-h_b, q) * f(-a+l_b, q)) #To make g's derivative 1 at l_b<x<h_b, multiply by 2/sqrt(q)

        #If within, f < 0 and near 0, gets larger the further away from the constraint rectangle
        rectangle_quality = 0.0001
        f_x = f_outside(x, rectangle_quality, x_center - width/2, x_center + width/2)
        f_y = f_outside(y, rectangle_quality, y_center - height/2, y_center + height/2)
        solver.subject_to(f_x + f_y > 0)

def SussyRect(solver: casadi.Opti, x, y, x0: float, y0: float, x1: float, y1: float, constrained_within: bool, n: int = 14):
    x_center = (x0 + x1)/2
    y_center = (y0 + y1)/2
    width = fabs(x0 - x1)/2  #x
    height = fabs(y0 - y1)/2 #y

    ellipsoid = (x - x_center) ** n * height ** n + (y - y_center) ** n * width **n
    denominator = width**n * height**n
    if constrained_within:
        solver.subject_to(ellipsoid < denominator)
    else:
        solver.subject_to(ellipsoid > denominator)

def particle(N) -> casadi.Opti:
    solver = casadi.Opti()
    solver.solver('ipopt')
    
    x_init = 0
    y_init = 0

    x_final = 10
    y_final = 10

    acc_max = 0.25

    X = solver.variable(4, N + 1)
    U = solver.variable(2, N)
    T = solver.variable(1)

    solver.subject_to(X[0, 0] == x_init)
    solver.subject_to(X[1, 0] == y_init)
    solver.subject_to(X[2, 0] == 0)
    solver.subject_to(X[3, 0] == 0)

    solver.subject_to(X[0, N] == x_final)
    solver.subject_to(X[1, N] == y_final)
    solver.subject_to(X[2, N] == 0)
    solver.subject_to(X[3, N] == 0)

    solver.subject_to(solver.bounded(-acc_max, U[:, :], acc_max))

    solver.subject_to(T > 0)
    dT = T/N
    for k in range(N):
        solver.subject_to(X[0, k + 1] == X[0, k] + X[2, k] * dT)
        solver.subject_to(X[1, k + 1] == X[1, k] + X[3, k] * dT)
        solver.subject_to(X[2, k + 1] == X[2, k] + U[0, k] * dT)
        solver.subject_to(X[3, k + 1] == X[3, k] + U[1, k] * dT)
        #SussyRect(solver, X[0, k], X[1, k], 3, 6, 6, 3, False) #big middle constraint
        if(k < N - 1):
            interX = (X[0, k] + X[0, k + 1])/2
            interY = (X[1, k] + X[1, k + 1])/2 
            ConstraintRectangle(solver, interX, interY, -.5, -.5, 12, 12, True)

            SmoothConstraintRectangle(solver, interX, interY, 2, 2, 3, 3, False)
            SmoothConstraintRectangle(solver, interX, interY, 8, 8, 9, 11, False)
            SmoothConstraintRectangle(solver, interX, interY, 8, 8, 11, 9, False)

        SmoothConstraintRectangle(solver, X[0, k], X[1, k], -.5, -.5, 12, 12, True)

        SmoothConstraintRectangle(solver, X[0, k], X[1, k], 2, 2, 3, 3, False)
        SmoothConstraintRectangle(solver, X[0, k], X[1, k], 8, 8, 9, 11, False)
        SmoothConstraintRectangle(solver, X[0, k], X[1, k], 8, 8, 11, 9, False)
    solver.minimize(T)
        

    return solver, X, U, T

if __name__ == "__main__":
    N = 50
    solver, X, U, T = particle(N)

    solution = solver.solve()
    for i in range(N + 1):
        states = solution.value(X)[0:2, i]
        print("(", states[0], ", ", states[1], ")")
        print("hehe")
        print("heha")
        print("hehe")
        print("heha")
        # f_x = fmax(fabs(states[0] - 4.5) - 3/2, 0)
        # f_y = fmax(fabs(states[1] - 4.5) - 3/2, 0)
        # print(f_x + f_y)