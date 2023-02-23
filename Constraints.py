from dataclasses import dataclass
from casadi import numpy as np
from casadi import *
import casadi

@dataclass
class ConstraintParameter:
    x0: float
    y0: float
    x1: float
    y1: float
    constrained_within: bool


def SmoothConstraintRectangle(solver: casadi.Opti, x, y, params: ConstraintParameter):
    x_center = (params.x0 + params.x1)/2
    y_center = (params.y0 + params.y1)/2
    width = fabs(params.x0 - params.x1)  #x
    height = fabs(params.y0 - params.y1) #y

    if params.constrained_within:
        f_inside = lambda a, l_b, h_b: -(a - l_b) * (a - h_b)

        f_x = f_inside(x, x_center - width/2, x_center + width/2)
        f_y = f_inside(y, y_center - height/2, y_center + height/2)
        solver.subject_to(f_x > 0)
        solver.subject_to(f_y > 0)
    else:
        f = lambda a, q: (sqrt(a**2 + q) + a - sqrt(q))/2
        f_outside = lambda a, q, l_b, h_b: -(f(a-h_b, q) * f(-a+l_b, q)) #To make g's derivative 1 at l_b<x<h_b, multiply by 2/sqrt(q)

        #If within, f < 0 and near 0, gets larger the further away from the constraint rectangle
        rectangle_quality = 0.0001
        f_x = f_outside(x, rectangle_quality, x_center - width/2, x_center + width/2)
        f_y = f_outside(y, rectangle_quality, y_center - height/2, y_center + height/2)
        solver.subject_to(f_x + f_y > 0)

def within_constraint(x, y, constraint: ConstraintParameter) -> bool:
    bigger_xconstraint = constraint.x0 if constraint.x0 > constraint.x1 else constraint.x1
    smaller_xconstraint = constraint.x0 if constraint.x0 < constraint.x1 else constraint.x1

    bigger_yconstraint = constraint.y0 if constraint.y0 > constraint.y1 else constraint.y1
    smaller_yconstraint = constraint.y0 if constraint.y0 < constraint.y1 else constraint.y1

    if constraint.constrained_within:
        return smaller_xconstraint < x < bigger_xconstraint and smaller_yconstraint < y < bigger_yconstraint
    else:
        return not(smaller_xconstraint < x < bigger_xconstraint) or not(smaller_yconstraint < y < bigger_yconstraint)