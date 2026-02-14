from math import sqrt

h = 1e-4
omega = 1.1
maxX = 6 * sqrt(10.0)

X = [0.0]
while X[-1] < maxX:
    X.append(X[-1] + h)
    h *= omega

print(len(X))
