import numpy as np
import math as m
import matplotlib.pyplot as plt
np.set_printoptions(formatter={'float_kind':'{:10.17f}'.format})

def func(x):
    return np.power(x,2) * np.sin(x)
start_point = -3
end_point = 4
n = 10
x= np.linspace(-m.pi,m.pi,n)
points = x.tolist()

def table(func, x):
    y = func(x)
    return  y

def polin(a, b, c, d, x, dot):
    return a + b*(x - dot) + c*(x - dot)**2 + d*(x - dot)**3

def splines(func, points):
    y = func(points)
    h = points[1] - points[0]
    a = np.zeros(len(y))
    a = y[:-1]
    b = np.zeros(len(y)-1)
    c = np.zeros(len(y)-1)
    d = np.zeros(len(y)-1)
    C = np.zeros((len(y)-1, len(y)-1))
    vect = np.zeros(len(y)-1)
    C[0][0] = 1
    C[len(y)-2][len(y)-2] = 1
    for row in range(2, len(y)-1):
        vect[row-1] = 3*((y[row] - y[row-1])/h - (y[row-1] - y[row-2])/h)
        C[row-1, row-2] = h
        C[row-1, row-1] = 4*h
        C[row-1, row] = h
    C_inv = np.linalg.inv(C)
    c= np.dot(C_inv, vect)
    for i in range(0, len(y)-2):
        b[i] = (y[i+1] - y[i])/h - h/3*(c[i+1] + 2*c[i])
        d[i] = (c[i+1] - c[i])/(3*h)
    b[len(y)-2] = (y[len(y)-1] - y[len(y)-2])/h - 2 * h * c[len(y)-2]/3
    d[len(y)-2] = - c[len(y)-2]/(3 * h)
    x = np.array([-2, -1, 0.5, 1, 1.5, 2, 2.3, 2.7, 3, 4])
    y = table(func, x)
    y_int = []
    for i in x:
        p = points.copy()
        p = np.append(p, i)
        p = np.sort(p)
        j = p.tolist().index(i)
        y_int.append(polin(a[j-2], b[j-2], c[j-2], d[j-2], i, points[j-2]))   
    return y, y_int, a, b, c, d

def show_spl(points, func):
    k, y_int, a, b, c, d = splines(func, points)
    x = np.array([])
    x_zoom = np.array([])
    y = np.array([])
    eps = np.array([])
    for i in range(len(points)-1):
        t = np.linspace(points[i], points[i+1])
        y = np.append(y, polin(a[i], b[i], c[i], d[i], t, points[i]))
        print(a[i], b[i], c[i], d[i])
        x = np.append(x,t)
        if i == len(points) - 5:
            x_zoom = x
            y_zoom = y
    eps = abs(y - func(x))
    eps_zoom = abs(y_zoom - func(x_zoom))
    fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=1, ncols=3,
    figsize=(8, 4))
    ax1.plot(x,y,color='m')
    ax1.grid()
    ax2.plot(x,eps, color = 'b')
    ax2.grid()
    ax3.plot(x_zoom, eps_zoom, color = 'b')
    ax3.grid()
    plt.show()

def lagr(x, func, points):
    y_int = func(points)
    L = 0
    for i in range(len(points)):
        product = 1
        div=1
        for j in range(len(points)):
            if i != j:
                product *= (x - points[j])
                div*= points[i] - points[j]
        L+=y_int[i]*product/div
    return L

def coefficient(x, y):
    m = len(x)
    t = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k-1:m-1])/(t[k:m] - t[:m-k])
    return a

def newton(x, func, points):
    y = func(points)
    #y = [1, 3, 4, 2, 0]
    a = coefficient(points, y)
    n = len(points) - 1  
    p = a[n]
    for k in range(1, n + 1):
        p = a[n - k] + (x - points[n - k])*p
    return p

def show_polin(func, func_polin, points):
    x = np.linspace(points[0], points[-1])
    y = func_polin(x, func,points)
    y_int = func(points)
    print(y_int)
    eps = abs(y - func(x))
    fig, (ax1, ax2) = plt.subplots(
    nrows=1, ncols=2,
    figsize=(8, 4))
    ax1.plot(x,y,color='m')
    ax1.scatter(points, y_int)
    ax1.grid()
    ax2.plot(x,eps, color = 'b')
    ax2.grid()
    plt.show()

def main():
    print(points)
    splines(func, points)
    show_spl(points, func)
    show_polin(func, lagr, points)
    show_polin(func, newton, points)
    print(coefficient(points,func(points)))
    
if __name__=="__main__":
    main()