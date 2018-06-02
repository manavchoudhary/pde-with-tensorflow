import tensorflow as tf
import numpy as np
from scipy.linalg import block_diag
import math
import re
import pickle
import sys
import time
import datetime

sess = tf.Session()

N = 100
h = 1.0/(N+1)
n = N**2

# Creating matrix A #
J = 4*np.eye(N)
i,j = np.indices(J.shape)
J[j==i+1] = -1
J[j==i-1] = -1

I_upper = -1*np.diag(np.ones(N*(N-1)), N)
I_lower = -1*np.diag(np.ones(N*(N-1)), -N)
repeated_J = [J]*N
A = block_diag(*repeated_J)
A = A + I_lower + I_upper
A_mat = tf.convert_to_tensor(A, dtype=tf.float64, name='A_matrix')

# Creating vector b and Creating weights i.e boundary values #
# Approach 4, create only 4N weights and then add them to vector b using a for loop instead of reshaping
#'''
b = (h**2)*20*np.ones((N**2,1))
b_tf = tf.constant(b, dtype=tf.float64, name='b_vector')

# left_boundary_weights = tf.Variable(np.zeros(N), dtype=tf.float64, name='left_boundary')
left_boundary_weights = tf.Variable(np.zeros((N, 1)), dtype=tf.float64, name='left_boundary')
padding_vector_0 = tf.constant(np.zeros(((N-1)*N, 1)), name='padding_1')
b_mat =  b_tf + tf.concat([left_boundary_weights, padding_vector_0], 0)

# lower_boundary_weights = tf.Variable(np.zeros(N), dtype=tf.float64, name='lower_boundary')
lower_boundary_weights = tf.Variable(np.zeros((N, 1)), dtype=tf.float64, name='lower_boundary')
# upper_boundary_weights = tf.Variable(np.zeros(N), dtype=tf.float64, name='upper_boundary')
upper_boundary_weights = tf.Variable(np.zeros((N, 1)), dtype=tf.float64, name='upper_boundary')

for i in range(0, N):
    padding_vector_1 = tf.constant(np.zeros((i*N, 1)) )
    padding_vector_2 = tf.constant(np.zeros((N-2, 1)) )
    padding_vector_3 = tf.constant(np.zeros(((N-i-1)*N, 1)) )
    b_mat = b_mat + tf.concat([padding_vector_1, lower_boundary_weights[i:i+1], padding_vector_2, upper_boundary_weights[i:i+1], padding_vector_3], 0)

# right_boundary_weights = tf.Variable(np.zeros(N), dtype=tf.float64, name='right_boundary')
right_boundary_weights = tf.Variable(np.zeros((N, 1)), dtype=tf.float64, name='right_boundary')
padding_vector_4 = tf.constant(np.zeros(((N-1)*N, 1)), name='padding_2')
b_mat = b_mat + tf.concat([padding_vector_4, right_boundary_weights], 0)
#'''


# objective_function : 0.5*Sum_over_domain[h^2*{y(x) - y_desired(x)}^2] + alpha*0.5*Sum_over_boundary[h^2{u(x) - u_desired(x)}^2]
# y_desired(x) = 3 + 5*x1*(x1-1)*x2*(x2-1)
# u_desired(x) = 0
# alpha = 0.01
# Domain--- x1 :: [1, N], x2 :: [1, N]
# Boundary--- x1 :: [0, N+1], x2 :: [0, N+1]

#constraints:
# y(x) <= 3.5
# 0 <= u(x) <= 10

alpha = 0.01

domain =  np.array([[1,1]])
for j in range(1, N+1):
    for i in range(1, N+1):
        if(i==1 and j==1):
            pass
        else:
            domain = np.concatenate((domain, np.array([[i,j]])), axis=0)

domain = domain/float(N+1)

y_desired = 3.0 + 5.0*domain[:, 0]*(domain[:, 0] - 1)*domain[:, 1]*(domain[:, 1] - 1)
# print(y_desired)
y_desired_tf =  tf.convert_to_tensor(y_desired, dtype=tf.float64, name='y_desired')
y_actual = tf.matrix_solve(A_mat, b_mat, name='matrix_solver')

# u_desired is summed over only the boundary so it should be of shape 4*N + 4 but i took a N*N vector because the inner domain values will be zero for both u_desired and weights (i.e u_actual)
# so the inner domain doesn't contribute to loss and hence the loss is computed only on the boundary.
# Approach 2
#'''
u_desired = tf.constant(np.zeros((4*N, 1)), name='u_desired')
weights = tf.concat([lower_boundary_weights, left_boundary_weights, upper_boundary_weights, right_boundary_weights], 0)
#'''


# loss = tf.add(0.5*(h**2)*tf.square(tf.norm(y_desired_tf-y_actual)), 0.5*(h)*alpha*tf.square(tf.norm(u_desired-weights)), name='loss_function')
loss = tf.add(0.5*(h**2)*tf.reduce_sum(tf.square(tf.subtract(y_desired_tf, y_actual))), 0.5*(h**2)*alpha*tf.reduce_sum(tf.square(tf.subtract(u_desired, weights))), name='loss_function')

inequalities = []
# Approach 4
#'''
for i in range(N*N):
    inequalities.append(3.5 - y_actual[i, 0])
for weight in [lower_boundary_weights, left_boundary_weights, upper_boundary_weights, right_boundary_weights]:
    for i in range(N):
        inequalities.append(weight[i, 0] - 0.0)
for weight in [lower_boundary_weights, left_boundary_weights, upper_boundary_weights, right_boundary_weights]:
    for i in range(N):
        inequalities.append(10.0 - weight[i, 0])
#'''


tf_graph_file_name = 'tf_graph'
writer = tf.summary.FileWriter(tf_graph_file_name, sess.graph)

# optimizer = tf.train.GradientDescentOptimizer(0.001)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, name='Adam')
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam')
# train_op = optimizer.minimize(loss)



# [(gradients, var)] = optimizer.compute_gradients(loss)
# train_op = optimizer.apply_gradients([(gradients*trainable_mask_mat, var)])
# for i in range(100):
#     _ = sess.run(train_op)


# optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, name='main_Q_network_adam_opt')
# minimize_op = optimizer.minimize(loss)
optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, equalities=[], inequalities=inequalities, method='SLSQP', options={'disp':True, 'iprint': 1})
# optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, equalities=[], inequalities=inequalities, options={'maxiter': 100000}, method='SLSQP')
# optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, equalities=[], inequalities=inequalities, method='COBYLA')
# optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, equalities=[], inequalities=inequalities, options={'maxiter': 100000}, method='COBYLA')
# optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, equalities=[], inequalities=inequalities, method='TNC')
# optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, equalities=[], inequalities=inequalities, options={'maxiter': 100000}, method='TNC')
# optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, equalities=[], inequalities=inequalities, method='L-BFGS-B')
# optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, equalities=[], inequalities=inequalities, options={'maxiter': 100000}, method='L-BFGS-B')

var_init = tf.global_variables_initializer()
sess.run(var_init)

start_time = time.time()
optimizer.minimize(sess)
end_time = time.time()

# start_time = time.time()
# for i in range(50000):
#     sess.run(minimize_op)
# end_time = time.time()

print('Value of N is '+str(N))
print('Total time taken is '+str(end_time-start_time))
print('Total time taken is '+str(datetime.timedelta(seconds=end_time-start_time)))

y_actual_val = sess.run(y_actual)

b_tf_val = sess.run(b_tf)
# print('The original b_tf is ')
# print(b_tf_val)


b_mat_val = sess.run(b_mat)
# print('The b_mat in the end is:')
# print(b_mat_val)


weights_val = sess.run(weights)
# print('The boundary control values are:')
# print(weights_val)

loss_val = sess.run(loss)
print('The value of loss function is '+str(loss_val))

#Comparison with IpOpt Error calculation:::::
ipopt_results_val = pickle.load(open('N_'+str(N)+'_results/ipopt_result_domain_values.pickle', 'rb'))
ipopt_results_val = np.array(ipopt_results_val)
# print(ipopt_results_val)
# print(ipopt_results_val.shape)
# ipopt_results_mat =  tf.convert_to_tensor(ipopt_results_val, dtype=tf.float32)
ipopt_results_mat =  tf.convert_to_tensor(ipopt_results_val, dtype=tf.float64)
error = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(ipopt_results_mat, y_actual))))
error_val = sess.run(error)
relative_error = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(ipopt_results_mat, y_actual)))) / tf.sqrt(tf.reduce_sum(tf.square(ipopt_results_mat)))
relative_error_val = sess.run(relative_error)
# print('The error compared to IpOpt solution on the domain values is '+str(error_val))
print('The "relative" norm of the error compared to IpOpt solution on the domain values is '+str(relative_error_val))
# for i in range(N**2):
#     print(str(y_actual_val[i][0])+' '*4+str(ipopt_results_val[i][0]))
print('Tensorflow domain value results')
# for i in range(N**2):
#     print(str(y_actual_val[i][0]))
# print('IpOpt domain value results')
# for i in range(N**2):
#     print(str(ipopt_results_val[i][0]))

print('-'*80)
ipopt_results_boundary_values = pickle.load(open('N_'+str(N)+'_results/ipopt_result_boundary_values.pickle', 'rb'))
ipopt_results_boundary_mat =  tf.convert_to_tensor(ipopt_results_boundary_values, dtype=tf.float64)
error_boundary_values = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(weights, ipopt_results_boundary_mat))))
error_boundary_values_val = sess.run(error_boundary_values)
relative_error_boundary_values = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(weights, ipopt_results_boundary_mat)))) / tf.sqrt(tf.reduce_sum(tf.square(ipopt_results_boundary_mat)))
relative_error_boundary_values_val = sess.run(relative_error_boundary_values)
print('The "relative" norm of the error compared to IpOpt solution on the boundary values is '+str(relative_error_boundary_values_val))
weights_val = sess.run(weights)
# for i in range(4*N):
#     print(str(weights_val[i][0])+' '*4+str(ipopt_results_boundary_values[i][0]))

print('Tensorflow boundary value results')

left_boundary_weights_values = sess.run(left_boundary_weights)
lower_boundary_weights_values = sess.run(lower_boundary_weights)
upper_boundary_weights_values = sess.run(upper_boundary_weights)
right_boundary_weights = sess.run(right_boundary_weights)
#Now combining the N*N and 4N values to create a (N+2)*(N+2) x 1 length vector.
combined_domain_and_boundary_values = np.zeros(((N+2)**2,1))
for j in range(0, N+2):
    for i in range(0, N + 2):
        if((j==0 and i==0) or (j==0 and i==N+1) or (j==N+1 and i==0) or (j==N+1 and i==N+1)):
            pass
        elif(j==0):
            combined_domain_and_boundary_values[i+(N+2)*j] = left_boundary_weights_values[i-1][0]
        elif(i==0):
            combined_domain_and_boundary_values[i+(N+2)*j] = lower_boundary_weights_values[j-1][0]
        elif(i==N+1):
            combined_domain_and_boundary_values[i+(N+2)*j] = upper_boundary_weights_values[j-1][0]
        elif(j==N+1):
            combined_domain_and_boundary_values[i+(N+2)*j] = right_boundary_weights[i-1][0]
        else:
            combined_domain_and_boundary_values[i+(N+2)*j] = y_actual_val[(i-1)+N*(j-1)][0]

file_name = 'N_'+str(N)+'_results/tf_optimization_both_domain_boundary_results.txt'
fp = open(file_name, 'w')
for i in range((N+2)**2):
    fp.write(str(combined_domain_and_boundary_values[i][0])+'\n')
fp.close()


# for i in range(4*N):
#     print(str(weights_val[i][0]))
# print('IpOpt boundary value results')
# for i in range(4*N):
#     print(str(ipopt_results_boundary_values[i][0]))

# lower_boundary_weights_val = sess.run(lower_boundary_weights)
# upper_boundary_weights_val = sess.run(upper_boundary_weights)
# left_boundary_weights_val = sess.run(left_boundary_weights)
# right_boundary_weights_val = sess.run(right_boundary_weights)
#
# final_tensorflow_results = left_boundary_weights_val
# for i in range(0, N**2):
#         final_tensorflow_results.append()
