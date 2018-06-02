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

N = 20
h = 1.0/(N+1)
n = N**2

J = 4*np.eye(N)
i,j = np.indices(J.shape)
J[j==i+1] = -1
J[j==i-1] = -1

I_upper = -1*np.diag(np.ones(N*(N-1)), N)
I_lower = -1*np.diag(np.ones(N*(N-1)), -N)
repeated_J = [J]*N
A = block_diag(*repeated_J)
A = A + I_lower + I_upper

A_mat = tf.convert_to_tensor(A, dtype=tf.float32)

b = (h**2)*20*np.ones((N**2,1))

fp = open('N_'+str(N)+'_results/ipopt_boundary_values.txt', 'r')
# fp = open('solution-USI.txt', 'r')
additional_values = {}

# print(additional_values)

for line in fp:
    line = line.strip()
    line = line.split('=')
    line[1] = float(line[1])
    row = re.search('\[(.+?),', line[0]).group(1)
    col = re.search(',(.+?)\]', line[0]).group(1)
    line[0] = row+col
    additional_values[line[0]] = line[1]

add_values = np.zeros((N**2,1))

N_num_digits = int(math.log10(N)) + 1

for j in range(1, N+1):
    for i in range(1, N+1):
        #Add boundary values for the bottom-left corner.
        if(i==1 and j==1):
            row = 0; col = 1
            index = ' '*(N_num_digits - 1) + str(row) + ' '*(N_num_digits - (int(math.log10(col))+1)) + str(col)
            add_values[(i - 1)+N*(j - 1)][0] += additional_values[index]
            row = 1; col = 0
            index = ' ' * (N_num_digits - (int(math.log10(row)) + 1)) + str(row) + ' ' * (N_num_digits - 1) + str(col)
            add_values[(i - 1)+N*(j - 1)][0] += additional_values[index]
        # Add boundary values for the top-left corner.
        elif(i==N and j ==1):
            row = N; col = 0
            index = ' ' * (N_num_digits - (int(math.log10(row)) + 1)) + str(row) + ' ' * (N_num_digits - 1) + str(col)
            add_values[(i - 1)+N*(j - 1)][0] += additional_values[index]
            row = N+1; col = 1
            index = ' ' * (N_num_digits - (int(math.log10(row)) + 1)) + str(row) + ' ' * (N_num_digits - (int(math.log10(col)) + 1)) + str(col)
            add_values[(i - 1)+N*(j - 1)][0] += additional_values[index]
        # Add boundary values for the bottom-right corner.
        elif (i == 1 and j == N):
            row = 0; col = N
            index = ' ' * (N_num_digits - 1) + str(row) + ' ' * (N_num_digits - (int(math.log10(col)) + 1)) + str(col)
            add_values[(i - 1)+N*(j - 1)][0] += additional_values[index]
            row = 1; col = N+1
            index = ' ' * (N_num_digits - (int(math.log10(row)) + 1)) + str(row) + ' ' * (N_num_digits - (int(math.log10(col)) + 1)) + str(col)
            add_values[(i - 1)+N*(j - 1)][0] += additional_values[index]
        # Add boundary values for the top-right corner.
        elif (i == N and j == N):
            row = N; col = N+1
            index = ' ' * (N_num_digits - (int(math.log10(row)) + 1)) + str(row) + ' ' * (N_num_digits - (int(math.log10(col)) + 1)) + str(col)
            add_values[(i - 1)+N*(j - 1)][0] += additional_values[index]
            row = N+1; col = N
            index = ' ' * (N_num_digits - (int(math.log10(row)) + 1)) + str(row) + ' ' * (N_num_digits - (int(math.log10(col)) + 1)) + str(col)
            add_values[(i - 1)+N*(j - 1)][0] += additional_values[index]
        # Add boundary values for the bottom row of domain.
        elif (i == 1):
            row = 0; col = j
            index = ' ' * (N_num_digits - 1) + str(row) + ' ' * (N_num_digits - (int(math.log10(col)) + 1)) + str(col)
            add_values[(i - 1)+N*(j - 1)][0] += additional_values[index]
        # Add boundary values for the leftmost column of domain.
        elif (j == 1):
            row = i; col = 0
            index = ' ' * (N_num_digits - (int(math.log10(row)) + 1)) + str(row) + ' ' * (N_num_digits - 1) + str(col)
            add_values[(i - 1)+N*(j - 1)][0] += additional_values[index]
        # Add boundary values for the top row of domain.
        elif (i == N):
            row = N+1; col = j
            index = ' ' * (N_num_digits - (int(math.log10(row)) + 1)) + str(row) + ' ' * (N_num_digits - (int(math.log10(col)) + 1)) + str(col)
            add_values[(i - 1)+N*(j - 1)][0] += additional_values[index]
        # Add boundary values for the rightmost column of domain.
        elif (j == N):
            row = i; col = N+1
            index = ' ' * (N_num_digits - (int(math.log10(row)) + 1)) + str(row) + ' ' * (N_num_digits - (int(math.log10(col)) + 1)) + str(col)
            add_values[(i - 1)+N*(j - 1)][0] += additional_values[index]

b = b + add_values
b_mat = tf.convert_to_tensor(b, dtype=tf.float32)

x_mat = tf.matrix_solve(A_mat, b_mat)
ipopt_results_val = pickle.load(open('N_'+str(N)+'_results/ipopt_result_domain_values.pickle', 'rb'))
ipopt_results_mat =  tf.convert_to_tensor(ipopt_results_val, dtype=tf.float32)

error = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(ipopt_results_mat, x_mat))))

start_time = time.time()
x_mat_val = sess.run(x_mat)
end_time = time.time()
error_val = sess.run(error)

#Now combining the N*N and 4N values to create a (N+2)*(N+2) x 1 length vector.
combined_domain_and_boundary_values = np.zeros(((N+2)**2,1))
for j in range(0, N+2):
    for i in range(0, N + 2):
        if((j==0 and i==0) or (j==0 and i==N+1) or (j==N+1 and i==0) or (j==N+1 and i==N+1)):
            pass
        elif((j==0 and i==1) or (j==0 and i==N)):
            combined_domain_and_boundary_values[i+(N+2)*j] = add_values[(i-1)+N*j][0]/2.0
        elif((j==1 and i==0) or (j==N and i==0)):
            combined_domain_and_boundary_values[i+(N+2)*j] = add_values[i+N*(j-1)][0]/2.0
        elif((j==1 and i==N+1) or (j==N and i==N+1)):
            combined_domain_and_boundary_values[i+(N+2)*j] = add_values[(i-2)+N*(j-1)][0]/2.0
        elif((j==N+1 and i==1) or (j==N+1 and i==N)):
            combined_domain_and_boundary_values[i+(N+2)*j] = add_values[(i-1)+N*(j-2)][0]/2.0
        elif(j==0):
            combined_domain_and_boundary_values[i+(N+2)*j] = add_values[(i-1)+N*j][0]
        elif(i==0):
            combined_domain_and_boundary_values[i+(N+2)*j] = add_values[i+N*(j-1)][0]
        elif(i==N+1):
            combined_domain_and_boundary_values[i+(N+2)*j] = add_values[(i-2)+N*(j-1)][0]
        elif(j==N+1):
            combined_domain_and_boundary_values[i+(N+2)*j] = add_values[(i-1)+N*(j-2)][0]
        else:
            combined_domain_and_boundary_values[i+(N+2)*j] = x_mat_val[(i-1)+N*(j-1)][0]

file_name = 'N_'+str(N)+'_results/forward_pde_both_domain_boundary_results.txt'
fp = open(file_name, 'w')
for i in range((N+2)**2):
    fp.write(str(combined_domain_and_boundary_values[i][0])+'\n')
fp.close()


print('Total time taken is '+str(end_time-start_time))
print('Total time taken is '+str(datetime.timedelta(seconds=end_time-start_time)))
# for i in range(len(x_mat_val)):
#     print(x_mat_val[i][0])
print('The overall error is'+str(error_val))

# for i in range(N**2):
#     print(str(x_mat_val[i][0])+' '*4+str(ipopt_results_val[i][0]))

