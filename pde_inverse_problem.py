import numpy as np
from burger_solver import get_burger_solver_periodic

def solve_burgers(alpha,log_kappa,a,b):
    kappa = np.exp(log_kappa)
    x_grid,solver = get_burger_solver_periodic(alpha,kappa,k = 0.001,n = 5000)
    def u0_fun(x):
        return a*np.sin(6*np.pi*x)- b*np.cos(2*np.pi*x) - 1
    u0 = u0_fun(x_grid)
    u_sol,tvals = solver(u0,final_t = 1.)
    return tvals,x_grid,u_sol

rng = np.random.default_rng(2)

alpha_range = (0,1)
log_kappa_range = (-6,0)
a_range = (0,5.)
b_range = (0,5)
true_params = np.array([0.5,-4.5,2,3.])
tvals,x_grid,u_true = solve_burgers(*true_params)
eval_locs = rng.uniform(0.02,0.98,(25,2))
u_observed = u_true(eval_locs[:,0],eval_locs[:,1],grid = False)

def loss(params):
    alpha,log_kappa,a,b = params
    tvals,x_grid,u_sol = solve_burgers(alpha,log_kappa,a,b)
    uval = u_sol(eval_locs[:,0],eval_locs[:,1],grid = False)
    return np.sum((uval - u_observed)**2)/2


# plt.contourf(x_grid,tvals,u_true(tvals,x_grid,grid = True),100)
# plt.scatter(eval_locs[:,0],eval_locs[:,1])