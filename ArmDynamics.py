from casadi import numpy as np
from casadi import MX
from casadi import sin, cos, pi

from dataclasses import dataclass

@dataclass
class Joint:
    mass: float
    length: float
    inertia: float
    comdist: float

    min_angle: float
    max_angle: float
    max_torque: float

class Arm:
    #
    # q = [θ1, θ2]ᵀ rad
    # q̇ = [θ̇1, θ̇2]ᵀ rad/s
    # u = [F_1, F_2] N m
    # 
    # M(q)q̈ + C(q, q̇)q̇ = τ_g(q) + u
    # q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + u)
    #
    #Length            |  (meters)                   | The distance of each joint from its axis to the next joint
    #Mass              |  (kilograms)                | The weight of all rotating components associated with a joint
    #Center Of Mass    |  (meters)                   | The distance of the center of mass from the joint's control axis
    #Moment Of Inertia |  (kilograms meters squared) | The inertia of all rotating components associated with a joint (about the control axis)
    #Max/Min           |  (radians)                  | The bounds for the valid states of the arm
    #Torque            |  (newton meters)            | The maximum and minimum applied torque for each joint
    #
    g = 9.806
    def __init__(self, proximal: Joint, distal: Joint):
        self.proximal = proximal
        self.distal = distal
    def MassMatrix(self, x):
        q = x[0:2, :]
        # F(thetas) = B(q)@u + C(qdot, q) + g(q)
        M = MX(2, 2) #Mass matrix
        M[0, 0] = self.proximal.inertia + self.proximal.length ** 2 * (self.proximal.mass + 2 * self.distal.mass)
        M[0, 1] = self.proximal.length * self.distal.length * self.distal.mass * cos(q[0, 0] - q[1, 0])
        M[1, 0] = self.proximal.length * self.distal.length * self.distal.mass * cos(q[0, 0] - q[1, 0])
        M[1, 1] = self.distal.inertia + self.distal.length ** 2 * self.distal.mass
    
        #Invert Mass Matrix
        Minv = MX(2, 2)
        Minv[0, 0] = M[1, 1]
        Minv[0, 1] = -M[0, 1]
        Minv[1, 0] = -M[1, 0]
        Minv[1, 1] = M[0, 0]
        detM = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        Minv /= detM
    
        return M, Minv
    def CoriolisMatrix(self, x):
        q = x[0:2]
        qdot = x[2:4]
    
        C = MX(2, 1) #Joint Velocity Product Term (Has qdot0**2/qdot1**2 Centripetal terms, but No qdot0*qdot1 Coriolis terms)
        C[0, 0] = self.proximal.length * self.distal.length * self.distal.mass * qdot[1, 0]**2 * sin(q[0, 0] - q[1, 0])
        C[1, 0] = -self.proximal.length * self.distal.length * self.distal.mass * qdot[0, 0]**2 * sin(q[0, 0] - q[1, 0])
        return C
    
    def GravityMatrix(self, x):
        q = x[0:2]
    
        G = MX(2, 1) #Gravity Term
        G[0, 0] = -self.g * (self.proximal.comdist * self.proximal.mass * cos(q[0, 0]) + self.distal.length * self.distal.mass * cos(q[0, 0]))
        G[1, 0] = -self.g * (self.distal.comdist * self.distal.mass * cos(q[1, 0]))
        return G
    
    def StabilizingTorques(self, x):
        return - self.GravityMatrix(x) - self.CoriolisMatrix(x)
    def StabilizedDynamics(self, x, u):
        q = x[0:2]
        qdot = x[2:4]
        M, Minv = self.MassMatrix(x)
    
        #Return changes in state:
        qddot = MX(4, 1)
        qddot[0:2] = qdot
        qddot[2:4] = Minv @ u
        return qddot
    def ArmDynamics(self, x, u):
        qdot = x[2:4]
        M, Minv = self.MassMatrix(x)
        C = self.CoriolisMatrix(x)
        G = self.GravityMatrix(x)
        #Return changes in state:
        qddot = MX(4, 1)
        qddot[0:2] = qdot
        qddot[2:4] = Minv @ (u - C + G)
    
        return qddot

def Prototype() -> Arm:
    pounds_to_kilograms = 0.453592
    inches_to_meters = 0.0254
    lbs_sqinches_to_kg_sqmeters = pounds_to_kilograms * inches_to_meters ** 2

    proximal = Joint(mass= 1.6 * pounds_to_kilograms, length= 17.5 * inches_to_meters, comdist= 9.88 * inches_to_meters, inertia = 236.59 * lbs_sqinches_to_kg_sqmeters, min_angle= 20 * (pi/180), max_angle= 160 * (pi/180), max_torque = 30)
    distal = Joint(mass= 0.88 * pounds_to_kilograms, length= 14.5 * inches_to_meters, comdist= 2.15 * inches_to_meters, inertia = 25.01 * lbs_sqinches_to_kg_sqmeters, min_angle= -240 * (pi/180), max_angle= 60 * (pi/180), max_torque = 15)
    
    return Arm(proximal, distal)

def Hogfish() -> Arm:
    pounds_to_kilograms = 0.453592
    inches_to_meters = 0.0254
    lbs_sqinches_to_kg_sqmeters = pounds_to_kilograms * inches_to_meters ** 2

    proximal = Joint(mass= 4*2*pounds_to_kilograms, length= 40*inches_to_meters, comdist= 22.8*inches_to_meters, inertia = 2*2961.95 * lbs_sqinches_to_kg_sqmeters, min_angle= 20 * (pi/180), max_angle= 160 * (pi/180), max_torque = 80)
    distal = Joint(mass= 2.7*2*pounds_to_kilograms, length= 33*inches_to_meters, comdist= 13.56*inches_to_meters, inertia = 2*866.84 * lbs_sqinches_to_kg_sqmeters, min_angle= -240 * (pi/180), max_angle= 60 * (pi/180), max_torque = 50)
    
    return Arm(proximal, distal)