import numpy as np
 
def bind360(x):
    #set within +pi -pi
    while x > np.pi:
        x -= 2 * np.pi
    while x < -np.pi:
        x += 2 * np.pi
    return x
def forwardKinematics(q, length_1, length_2):
    q1 = None
    q2 = None
    if(q.ndim == 2):
        q1 = q[0, 0]
        q2 = q[1, 0]
    elif(q.ndim == 1):
        q1 = q[0]
        q2 = q[1]
    x = np.empty((2, 1))
    x[0, 0] = np.cos(q1)*length_1+np.cos(q2)*length_2
    x[1, 0] = np.sin(q1)*length_1+np.sin(q2)*length_2
    return x

def inverseKinematics(x, length_1, length_2):
    pX = None
    pY = None
    if(x.ndim == 2):
        pX = x[0, 0]
        pY = x[1, 0]
    elif(x.ndim == 1):
        pX = x[0]
        pY = x[1]
    targetDist = np.hypot(pX, pY)
    if(np.abs(pX) + np.abs(pY) > 0 and length_1 != length_2):
        pX += 0.001
        targetDist = np.hypot(pX, pY)
    if(targetDist > length_1 + length_2):
        targetDist = (length_1 + length_2 - 0.001)
    if(targetDist < np.abs(length_1 - length_2)):
        targetDist = (np.abs(length_1 - length_2) + 0.001)
 
    signAngle = 1 if pX > 0 else -1
    angle1 = np.arctan2(pY, pX) + signAngle * np.arccos((targetDist**2 + length_1**2 - length_2**2)/(2 * length_1 * targetDist))
    angle2 = signAngle * (np.arccos((length_1**2 + length_2**2 - targetDist**2)/(2 * length_2 * length_1)))

    angle1 = bind360(angle1)
    angle2 = bind360(angle1 + angle2 - np.pi)
    if angle2 > np.pi/2: angle2 -= 2 * np.pi
    return np.array([angle1, angle2])
 
def inverseVelocities(q, xdot, length_1, length_2):
    #dxi/dqi
    epsilon = 1/(2.0 ** 10)
    J = np.empty((2, 2))
    for i in range(2):
        right = q.copy()
        left = q.copy()
        right[i] = right[i] + epsilon
        left[i] = left[i] - epsilon
        diff = ((forwardKinematics(right, length_1, length_2) - forwardKinematics(left, length_1, length_2))/(2 * epsilon))[:, 0]
        for j in range(2):
            J[j, i] = diff[j]
    return np.linalg.inv(J) @ xdot