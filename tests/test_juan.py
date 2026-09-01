import numpy as np
import matplotlib.pyplot as plt

def f(x):
    x0 = x[0]
    x1 = x[1]
    y = 100.0 * (x1 - x0*x0)*2 + (x0 - 1)*2 
    return y

def grad(x):
    x0 = x[0]
    x1 = x[1]
    gx0 = 200.0*(x1 - x0*x0)*(-2.0*x0) + 2.0*(x0-1.0)
    gx1 = 200.0*(x1 - x0*x0)
    return np.array([gx0, gx1])

def line_search(x, p, g):
    c1 = 1e-4
    fx = f(x)
    gTp0 = np.dot(g, p)
    alpha = 1.0
    for it in range(30):
        x_new = x + alpha * p
        f_new = f(x_new)

        armijo = f_new <= fx + c1 * alpha * gTp0

        print("\n--- line_search_lo_hi ---")
        print("it =", it)
        print("alpha =", alpha)
        print("f0 =", fx)
        print("f_new =", f_new)
        print("gTp0 =", gTp0)
        print("Armijo =", armijo)

        if armijo:
            return alpha

        alpha *= 0.5 
    return alpha

ni = 50
x0 = 0.5
z0 = -0.5
x = np.array([x0, z0])
s_store = []
y_store = []
history = []
for _ in range(ni):
    g = grad(x)
    g = g/np.max(np.abs(g))
    history.append((x[0], x[1]))
    p = -g
    alpha = line_search(x,p,g)
    x_new = x + alpha * p
    g_new = grad(x_new)
    g_new = g_new/np.max(np.abs(g_new))

    x = x_new.copy()

history.append((x[0], x[1]))
history = np.array(history)

x1 = np.linspace(-2.0, 2.0, 400)
x2 = np.linspace(-2.0, 2.0, 400)
X1, X2 = np.meshgrid(x1, x2)
F = f([X1, X2])

plt.figure()
plt.contourf(X1, X2,F, levels=30)
plt.plot(history[:,0], history[:,1], 'r.-', label='trajetória')
plt.plot(1, 1, 'bo', label='mínimo (1,1)')
plt.xlabel("x")
plt.ylabel("z")
plt.title("Rosenbrock")
plt.legend()
plt.grid()
plt.show()
