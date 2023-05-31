import matplotlib.pyplot as plt
import numpy as np
import math

def f(x, y):
    return x + 2 * y + 10 * (x ** 2) - 4 * x * y + 10 * (y ** 2)

def grad_f(x, y):
    return 20*x - 4*y + 1, 20*y - 4*x + 2



def descente(Points, lamb=[0.001,0.002,0.003,0.004,0.09], eps=1e-5, maxIter=2000):
    minValue = -0.1510416
    error = np.array([0.0002, 0.0005, 0.001])
    opt = np.empty((0, 3))
    tempi = []
    count = 1
    for xy in Points:
        print("Point " + str(count)+ "\n")
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
            x = x - lamb[4] * gradx #changeable lamda values
            y = y - lamb[4] * grady #changeable lamda values
            i += 1
            total = f(x , y)
            Optimalitygap = total - minValue
            print("i=%s  x=%.7f  y=%.7f  gradx=%.7f  grady=%.7f Total=%.7f Optim.Gap=%.7f" % (i, x, y, gradx, grady, total, Optimalitygap))
            if Optimalitygap <= error[0] and not ch1:
                tempopt = np.append(tempopt, i)
                ch1 = True
            elif Optimalitygap <= error[1] and not ch2:
                tempopt = np.append(tempopt, i)
                ch2 = True
            elif Optimalitygap <= error[2] and not ch3:
                tempopt = np.append(tempopt, i)
                ch3 = True
        tempopt = tempopt[::-1]
        tempopt = tempopt.reshape(1, -1)
        opt = np.concatenate((opt, tempopt), axis=0)
        count += 1
        print("\n")

    
    plt.plot(error, opt[0, :], 'g', label='Point 1')
    plt.plot(error, opt[0, :], 'g*')
    plt.plot(error, opt[1, :], 'b', label='Point 2')
    plt.plot(error, opt[1, :], 'b*')
    plt.plot(error, opt[2, :], 'r', label='Point 3')
    plt.plot(error, opt[2, :], 'r*')
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
        plt.plot(lamb, tempi, 'r*')
        plt.xlabel('Lambda')
        plt.yscale('linear')
        plt.xscale('linear')
        plt.ylabel('Number of iterations')
        plt.title("Point " + "("+ str(xy[0]) + ";" + str(xy[1]) + ") (Lambda vs Number of iterations)" )
        for test in zip(lamb, tempi):
            plt.annotate('(%.4f, %.4f)' % test, xy=xy)
        plt.show()



x_values = [23,11,-5]
y_values = [4,21,-80]

Points = [[x, y] for x, y in zip(x_values, y_values)]

descente(Points)


print("Repository: https://github.com/Ohmanitsmebruh/Assignment10.git \n")
input("Execution finished! Press any key to exit.")