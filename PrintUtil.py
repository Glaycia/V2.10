from casadi import sin, cos

def printArm(q, length1, length2, timestep):
    ox = 0
    oy = 0
    x1 = ox + cos(q[0]) * length1
    y1 = oy + sin(q[0]) * length1
    x2 = x1 + cos(q[1]) * length2
    y2 = y1 + sin(q[1]) * length2

    ox = rawnumber(ox)
    oy = rawnumber(oy)
    x1 = rawnumber(x1)
    y1 = rawnumber(y1)
    x2 = rawnumber(x2)
    y2 = rawnumber(y2)

    printline(ox, x1, oy, y1, timestep)
    printline(x1, x2, y1, y2, timestep)
    print("(", x2, ", ", y2, ")", "\\left\\{n\\ge", timestep, "\\right\\}")
    print("0\n0")
def rawnumber(x):
    return ("%.17f" % x).rstrip('0').rstrip('.')
def printline(x1, x2, y1, y2, timestep):
    print("((1-t)(", x1,")+t(", x2, "),(1-t)(", y1,")+t(", y2, "))\\left\\{n\\ge", timestep, "\\right\\}")