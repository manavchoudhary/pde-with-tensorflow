# pde-with-tensorflow
Implements elliptic partial differential equations with Dirichlet boundary conditions in Tensorflow

1. Uses tf.contrib.opt.ScipyOptimizerInterface with 'SLSQP' method for constrained optimization of the reverse problem.
2. Uses tf.matrix_solve to solve the forward PDE problem.
