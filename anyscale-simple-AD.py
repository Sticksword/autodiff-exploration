import autodiff as ad
import numpy as np

def add(a, b):
  if isinstance(a, str):
    a = ad.Variable(name = a)
  if isinstance(b, str):
    b = ad.Variable(name = b)  
  return a + b

def multiply(a, b):
  if isinstance(a, str):
    a = ad.Variable(name = a)
  if isinstance(b, str):
    b = ad.Variable(name = b)  
  return a * b

def evaluate(expr, value_map):
  """Evaluate expr for a specific value of x.

  Examples:
    >>> evaluate(multiply(3, 'x'), {'x': 0})
        0 # because 3*0 = 0
    >>> evaluate(add(3, multiply('x', 'x')), {'x': 2})
        7 # because 3+2^2 = 7

  Input:
    expr: An expression. The expression can be the following
      (1) any real number
      (2) 'x'
      (3) operation(expr, expr), where operation can be either add or multiply
    value_map: A dictionary specifying the values of the variables.
  
  Output:
    This returns expr evaluated at the specific value of x, which should be a
    real number.
  """
  # TODO: Implement this.
  y = expr
  # print(y)
  input_vars = [[ad.Variable(name = x_val), value_map[x_val]] for x_val in value_map.keys()]
  # print(input_vars)
  x_vars = [x_var for x_var, x_val in input_vars]
  # print(x_vars)
  # print(type(x_vars[0]))

  x_grads = ad.gradients(y, x_vars)

  executor = ad.Executor([y, *x_grads])

  feed_dict = {}
  for x, x_val in input_vars:
    feed_dict[x] = x_val * np.ones(1) # the AD code actually supports matrix multiplication but for our purposes we really just need scalar value support

  output_nodes = executor.run(feed_dict = feed_dict)
  y_val = output_nodes[0]

  return y_val

def differentiate(expr, value_map):
  """Compute derivative of expr with respect to x for a specific value of x.

  Examples:
    >>> differentiate(multiply(3, 'x'), {'x': 0})
        3 # because d(3x)/d(x) = 3
    >>> differentiate(add(3, multiply('x', 'x')), {'x': 2})
        4 # because d(3+x^2)/d(x) = 2x -> 2*2 = 4

  Input:
    expr: An expression. The expression can be the following
      (1) any real number
      (2) 'x'
      (3) operation(expr, expr), where operation can be either add or multiply
    value_map: A dictionary specifying the values of the variables.

  Output:
    This returns the derivative of expr with respect to x evaluated at the
    Specific value of x, which should be a real number.
  """
  # TODO: Implement this.
  y = expr
  input_vars = [[ad.Variable(name = x_val), value_map[x_val]] for x_val in value_map.keys()]
  x_vars = [x_var for x_var, x_val in input_vars]

  x_grads = ad.gradients(y, x_vars)

  executor = ad.Executor([y, *x_grads])

  feed_dict = {}
  for x, x_val in input_vars:
    feed_dict[x] = x_val * np.ones(1) # the AD code actually supports matrix multiplication but for our purposes we really just need scalar value support

  output_nodes = executor.run(feed_dict = feed_dict)
  print(output_nodes)

  return output_nodes[1: -1] # first output is y val, rest are gradient nodes

print('evaluate test case 1')
print(evaluate(multiply(3, 'x'), {'x': 0}))

print('evaluate test case 2')
print(evaluate(add(3, multiply('x', 'x')), {'x': 2}))

print('differentiate test case 1')
print(differentiate(multiply(3, 'x'), {'x': 0}))

print('differentiate test case 2')
n = 10
expr = 'x'
for _ in range(n):
  expr = add(expr, expr)
print(differentiate(expr, {'x': 1}))