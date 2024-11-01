
import numpy as np
np.set_printoptions(precision = 4)

x = np.array([[0,0], [0,1], [1,0], [1,1]]).T
print('input:\n',x)
print('shape:', x.shape)

y = np.array([[0,0,0,1]])
print('output:\n',y)
print('shape:', y.shape)

x[x == 0] = -1
y[y == 0] = -1
print('bipolar input:\n', x)
print('bipolar output:\n', y)

print('\n adding bias,\n')

A = np.vstack((np.ones((1, x.shape[1])), x))
print('input:\n', A)
print('shape:\n', A.shape)

print('\n iniatialize weight vector,\n')
w = np.zeros((A.shape[0], 1))
print('weight:\n', w)
print('shape:\n', w.shape)

MAX_EPOCH = 100
lr = 0.2

print('\n ADALINE NETWORK,\n')

Er = []
for ep in range(MAX_EPOCH):
    print('epoch:', ep+1, '='*80)
    error = []

    for ip, op in zip(A.T, y.ravel()):
        print('current input:', ip, end = '\t')
        print('weight:', w.T.ravel())
        y_pred = (w.T @ ip)[0]
        print('prediction = {0:5.4f}'.format(y_pred), sep='')

        w += lr*(op - y_pred)*ip.reshape(-1,1)
        error.append((op - y_pred)**2)

    Er.append(sum(error))
    if Er[-1] == 0:
      print('convergence at ', ep+1)

    if len(Er) > 2:
      if Er[-1] - Er[-2] < 0.00001:
        print('convergence at ', ep+1)
      break

print('optimum weights:\n', w.T.ravel())
y_pred = (w.T @ A).ravel()
print('prediction:\n', y_pred)
print('error:\n', Er)
print('predicted output:\n', y_pred)
