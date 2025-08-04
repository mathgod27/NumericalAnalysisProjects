
# Consider the following nonlinear function: f(x) = (x1 + 10*x2)**2 + 5*(x3 - x4)**4 + 10*(x1 - x4)**4
# the initial guess x0 = [3;-1;0;1], step length alpha = 0.001, and the convergence threshold is set to 0.01. 
#
# Modify the in-class code to preform the gradient descent method. Also need to state the iterations and the final x values at convergence.

#In class code:

# import the libraries
import numpy as np
import math

## a function returns the Gradient vector at the input vector x_i
## note: the expression of each element in the Gradient vector is hard-coded based on the formula presented above, which is also given on Slide 40 of Topic 3.1

#this part is modified for our function when using partial derivatives of each x1-4. g[0] corresponds to our partial derivative of x1, and etc
def getGradientVector(x_i):
  g = np.zeros((4,1))   #(amount of x_i, 1)
  g[0] = 2 * (x_i[0] + 10*x_i[1]) + 40*(x_i[0] - x_i[3])**3
  g[1] = 20 * (x_i[0] + 10 * x_i[1])
  g[2] = 20 * (x_i[2] - x_i[3])**3
  g[3] = -20 * (x_i[2] - x_i[3])**3 - 40 * (x_i[0] - x_i[3])**3
  return g

# a simple function returns the norm of an input vector
def getNorm(v):
  return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3])   #v[x_i]*v[x_i]


## set an initial guess
x_0 = np.array([[3.0], [-1.0], [0.0], [1.0]]) #we got this initial guess given in the problem
print(f"--The initial guess x0: ")
print(x_0)

## set a threshold value for getting a good solution (given in problem)
thresh = 0.01

## set step length (given in problem)
alpha = 0.001


#this whole part and onward is not edited 
x_i = x_0
count = 0
while True:
  print('****************************************************')
  print(f'--At iteration: {count}')
  
  ## compute the gradient vector at x_i
  g = getGradientVector(x_i)
  print(f"--The gradient vector g(x_{count}): ")
  print(g)

  ## compute the norm of the gradient vector at x_i
  normOfGradient = getNorm(g)
  print(f"--The norm of the gradient vector |g(x_{count})|: {normOfGradient}")

  ## check if converged; if true, then stop; continue otherwise
  if normOfGradient <= thresh:
    break


  ## get h=-g
  h_i = -1*g
  print(f"--The h vector h_{count}: ")
  print(h_i)

  ## get the new solution point
  x_i_plus1 = x_i + alpha * h_i
  print(f"--The solution vector x_{count+1}: ")
  print(x_i_plus1)

  x_i = x_i_plus1
  count = count + 1
  print()

