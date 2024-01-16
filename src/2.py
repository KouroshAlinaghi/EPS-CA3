import numpy as np, random
import matplotlib.pyplot as plt

green = '#57cc99'
red = '#e56b6f'
blue = '#22577a'
yellow = '#ffca3a'

bar_width = 0.5
line_thickness = 2

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
set_seed(810109203)

points = [(-2.3, -9.6), (-1.1, -4.9), (0.5, -4.1), (3.2, 2.7), (4.0, 5.9), (6.7, 10.8), (10.3, 18.9), (11.5, 20.5)]
special_points = [(5.8, 31.3), (20.4, 14.1), (20.4, 31.3)]
outlier, high_leverage, both = special_points

x_reg = [point[0] for point in points]
y_reg = [point[1] for point in points]

x_spe = [point[0] for point in special_points]
y_spe = [point[1] for point in special_points]

plt.plot(x_reg, y_reg, 'o', color=green, label='Normal points', alpha=0.7)
plt.plot(x_spe, y_spe, 'o', color=red, label='Special points', alpha=0.7)
plt.legend()
plt.grid(True)
plt.show()

def p2l_y_dist(point_x, point_y, m, c):
    y_pred = m * point_x + c
    return abs(point_y - y_pred)

def calc_err(inp_points, m, c):
    squares_error = 0
    for point in inp_points:
        squares_error += p2l_y_dist(point[0], point[1], m, c) ** 2
    return squares_error

def linear_regression(inp_points, l = -12, r = 12, steps = 0.10):
    ans = (0, 0, int(1e9))
    for m in range(int(l / steps), int(r / steps)):
        m *= steps
        for c in range(int(l / steps), int(r / steps)):
            c *= steps
            error = calc_err(inp_points, m, c)
            if error < ans[2]:
                ans = (m, c, error)
    return ans

points_sets = [
    points,
    points + [outlier],
    points + [high_leverage],
    points + [both]
]

titles = [
    'Normal points',
    'Normal points + outlier point',
    'Normal points + high_leverage point',
    'Normal points + outlier-high_leverage point',
]

for i in range(len(points_sets)):
    points_set = points_sets[i]
    x_pts = [point[0] for point in points_set]
    y_pts = [point[1] for point in points_set]

    x = np.linspace(np.min(x_pts) - 1, np.max(x_pts) + 1, 100)

    sst = np.sum((y_pts - np.mean(y_pts)) ** 2)
    m, c, ssr = linear_regression(points_set)
    y = m*x + c

    r_squared = 1 - ssr / sst

    line_eq_format = "y = {:.2g}x - {:.2g}".format(m, -c) if c < 0 else "y = {:.2g}x + {:.2g}".format(m, c)

    plt.figure() 
    plt.plot(x_pts, y_pts, 'o', color=green, alpha=0.7, label='Points')
    plt.plot(x, y, color=red, label=line_eq_format)
    plt.title(titles[i] + ", Coefficient of Determination = {:.5g}".format(r_squared))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()

