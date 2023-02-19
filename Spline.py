from casadi import numpy as np

class Spline:
    def __init__(self, waypoint_list, n_points):
        self.waypoints = waypoint_list

        self.waypoint_div = [0] * (int)(len(self.waypoints) - 1)

        for i in range(n_points - 1):
            self.waypoint_div[i%len(self.waypoint_div)] += 1
    
    def points(self):
        points = []
        for i in range(len(self.waypoints) - 1):
            start = self.waypoints[i]
            end = self.waypoints[i + 1]
            splinesection = Segment(start.p, start.v, start.a, end.p, end.v, end.a)
            points = points + splinesection.points(self.waypoint_div[i])

        points.append(self.waypoints[len(self.waypoints) - 1].p)

        return points

class Waypoint:
    p: np.array
    v: np.array
    a: np.array
    def __init__(self, p, v, a):
        self.p = p
        self.v = v
        self.a = a

class Segment:
    def __init__(self, p0: np.array, v0: np.array, a0: np.array, p1: np.array, v1: np.array, a1: np.array):
        self.parameters = np.empty((6, len(p0)))

        self.parameters[0] = p0
        self.parameters[1] = p1
        self.parameters[2] = v0
        self.parameters[3] = v1
        self.parameters[4] = a0
        self.parameters[5] = a1

        self.parameters = self.parameters.T

    def interpolate(self, t: float) -> np.array:
        splineform = np.array([
            1 + t**3 * (-10 + 15*t - 6 * t**2),
            t**3 * (10 - 15*t + 6 * t**2),
            t + t**3 * (-6 + 8*t - 3 * t**2),
            t**3 * (-4 + 7*t - 3 * t**2),
            t**2 * (0.5 - 1.5 * t + 1.5 * t**2 - 0.5 * t**3),
            t**3 * (0.5 - 1.0 * t + 0.5 * t**2)
        ])
        return (self.parameters @ splineform)[:]

    def points(self, N: int):
        points = []
        for i in range(N):
            points.append(self.interpolate(i/(N)))
        return points

if __name__ == "__main__":
    waypoints = [
        Waypoint(p=np.array([0, 0]),v=np.array([10, 0]), a=np.array([0, 0])),
        Waypoint(p=np.array([10, 1]),v=np.array([10, 0]), a=np.array([0, 0])),
        Waypoint(p=np.array([20, 0]),v=np.array([10, 0]), a=np.array([0, 0]))
    ]
    N = 50
    path = Spline(waypoints, N)
    points = path.points()
    for i in range(N):
        print("(", points[i][0], ",", points[i][1], ")")