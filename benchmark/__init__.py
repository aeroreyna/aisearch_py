import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

scaleInput = True
dimOnAxis = 1


def plot_3d(bm_function, steps=100):
    X = np.arange(steps)/(steps-1)
    Y = np.arange(steps)/(steps-1)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros([steps, steps])
    for x in range(steps):
        for y in range(steps):
            Z[x, y] = bm_function(np.array([x/(steps-1), y/(steps-1)]))
    # fig = plt.figure()
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=1, fontsize=5)
    plt.show(block=False)
    plt.pause(0.001)
    return ax


def schwefel(x):
    """
    Search domain: -500 < xi < 500, i = 1, 2, . . . , n.
    Number of local minima: several local minima.
    global minimum
    f(x) = 0
    x(i) = 420.90687, i=1:n
    """
    if scaleInput:
        x = x * 1000-500
    dim = x.shape[dimOnAxis] if len(x.shape) > 1 else x.shape[0]
    axis = 1 if len(x.shape) > 1 else 0
    return 418.9829 * dim + np.sum(-x * np.sin(np.sqrt(np.abs(x))), axis)
