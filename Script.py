import matplotlib.pyplot as plt
import numpy as np
import math

def f(x, y):
    return x + 2 * y + 10 * (x ** 2) - 4 * x * y + 10 * (y ** 2)

def grad_f(x, y):
    return 20*x - 4*y + 1, 20*y - 4*x + 2



def descente(Points, lamb=[0.001,0.002,0.003,0.004,0.005], eps=1e-10, maxIter=2000):
    minValue = -0.1510416
    error = np.array([0.0002, 0.0005, 0.001])
    opt = np.empty((0, 3))
    tempi = []
    for xy in Points:
        tempopt = np.array([])
        i = 0
        x = xy[0]
        y = xy[1]
        gradx, grady = grad_f(x, y)
        grad = math.sqrt(gradx ** 2 + grady ** 2)
        ch1 = False
        ch2 = False
        ch3 = False
        while abs(grad) > eps:
            gradx, grady = grad_f(x, y)
            grad = math.sqrt(gradx ** 2 + grady ** 2)
            x = x - lamb[0] * gradx
            y = y - lamb[0] * grady
            i += 1
            total = f(x , y)
            Optimizationgap = total - minValue
            
            if Optimizationgap <= error[0] and not ch1:
                tempopt = np.append(tempopt, i)
                ch1 = True
            elif Optimizationgap <= error[1] and not ch2:
                tempopt = np.append(tempopt, i)
                ch2 = True
            elif Optimizationgap <= error[2] and not ch3:
                tempopt = np.append(tempopt, i)
                ch3 = True
        tempopt = tempopt[::-1]
        tempopt = tempopt.reshape(1, -1)
        opt = np.concatenate((opt, tempopt), axis=0)

    
    plt.plot(error, opt[0, :], 'g', label='Point 1')
    plt.plot(error, opt[1, :], 'b', label='Point 2')
    plt.plot(error, opt[2, :], 'r', label='Point 3')
    plt.xlabel('Optimality Gap')
    plt.yscale('log')
    plt.ylabel('Number of iterations')
    plt.title("Optimality Gap vs Number of iterations")
    plt.legend()
    plt.show()

    for xy in Points:
        tempi= []
        for l in lamb:
            i = 0
            x = xy[0]
            y = xy[1]
            gradx, grady = grad_f(x, y)
            grad = math.sqrt(gradx ** 2 + grady ** 2)
            
            while abs(grad) > eps:
                gradx, grady = grad_f(x, y)
                grad = math.sqrt(gradx ** 2 + grady ** 2)
                x = x - l * gradx
                y = y - l * grady
                i += 1
                total = f(x , y)
                if i > maxIter:
                    break
            tempi.append(i)
        plt.plot(lamb, tempi)
        plt.xlabel('Lambda')
        plt.yscale('linear')
        plt.xscale('linear')
        plt.ylabel('Number of iterations')
        plt.title("Point " + "("+ str(xy[0]) + ";" + str(xy[1]) + ") (Lambda vs Number of iterations)" )
        plt.show()



x_values = []
y_values = []
for i in range(3):
    x = int(input("Point " + str(i+1) + ": x "))
    y = int(input("Point " + str(i+1) + ": y "))
    x_values.append(x)
    y_values.append(y)

Points = [[x, y] for x, y in zip(x_values, y_values)]

descente(Points)

input("Execution finished! Press any key to exit.")