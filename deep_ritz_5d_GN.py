import jax.numpy as jnp
import jax.numpy.linalg as alg
import jax.flatten_util
from jax import jacfwd, hessian
from jax import vmap, grad, jit, random

from tool.model import normal_init
from tool.model import shallow_network
from tool.quadrature import QuasiMonteCarlo
from tool.gauss_newton import gn_direction, grid_line_search
from tool.gauss_newton import jacobian_matrix, jacobian_matrix_multi_dim

# for more precisions
jax.config.update("jax_enable_x64", True)

# activation functions
tanh = lambda x: jnp.tanh(x)
relu2 = lambda x: jnp.where(x>0,x,0)**2
relu3 = lambda x: jnp.where(x>0,x,0)**3
relu4 = lambda x: jnp.where(x>0,x,0)**4

# quadrature for training and testing
quad_rule = QuasiMonteCarlo()
n_rectangle = jnp.array([[0.,1.],[0.,1.],[0.,1.],[0.,1.],[0.,1.]])
num_samples_for_train = 16000
num_samples_for_test = 20000
training_quadrature = quad_rule.n_rectangle_samples(n_rectangle, num_samples_for_train, K=30)
testing_quadrature = quad_rule.n_rectangle_samples(n_rectangle, num_samples_for_test, K=1)

# model, exact solution and source term.
# reshape(x,()) returns a single number, not an array object.

pi = jnp.pi
model = shallow_network(relu4)
u_exact  = lambda p: jnp.reshape(jnp.cos(pi*p[...,0:1]) + \
                     jnp.cos(pi*p[...,1:2]) + \
                     jnp.cos(pi*p[...,2:3]) + \
                     jnp.cos(pi*p[...,3:4]) + \
                     jnp.cos(pi*p[...,4:5]), ())
rhs = lambda p: jnp.reshape(2*pi**2 * u_exact(p), ())
v_model = vmap(model, (None, (0)))
v_u_exact = vmap(u_exact, (0))
v_rhs = vmap(rhs, (0))


# loss function of deep ritz methods
# for the equation -Delta(u) + u = f
# with Neumann's boundary du/dn = 0.

@jit
def loss_gradient(params):
    nabla_u = vmap(grad(lambda p: model(params, p)), (0))
    nabla_u_square = lambda p: 0.5 * jnp.reshape(jnp.sum(nabla_u(p)**2,axis=1), (len(p),))
    return training_quadrature(nabla_u_square)

@jit
def loss_linear_term(params):
    u_square = lambda p: 0.5 * (jnp.pi**2) * jnp.reshape(v_model(params, p)**2, (len(p),))
    return training_quadrature(u_square)

@jit 
def loss_rhs(params):
    u_dot_f = lambda p: jnp.reshape(v_model(params, p)*v_rhs(p), (len(p),))
    return training_quadrature(u_dot_f)

@jit
def loss(params):
    return loss_gradient(params) + loss_linear_term(params) - loss_rhs(params)


# two types of jacobian concerning the
# loss function of -Delta(u) + u = f.
# dt: derivative for theta.
# di: gradient for spacial variable(s).

def dt_di_model(model):
    
    def dt_model(params, p):
        return grad(model, (0))(params, p)
    
    def dtdi_single_input(params, p):
        return jacfwd(dt_model, (1))(params, p)
    
    return dtdi_single_input  

def dt_model(model):
    
    def dt_single_input(params, p):
        return grad(model, (0))(params, p)
    
    return dt_single_input  


d1_model = dt_model(model)
d2_model = dt_di_model(model)
    
def jac_2nd_elliptic(params):
    """
        The Jacobian matrix for the escond order 
        elliptic equation. In this case, the coefficients 
        in front of -Delta(u) is set to be pi^2. This 
        function can be used as a handle that takes 
        "params" as input.
    """
    d1_jacobian = jacobian_matrix(d1_model, training_quadrature)
    d2_jacobian = jacobian_matrix_multi_dim(d2_model, training_quadrature)

    return (jnp.pi**2)*d2_jacobian(params) + d1_jacobian(params)

# get the jacobian generator for the equation and the 
# gauss-newton-direction solver.
# example: 
# -------- print(direction(params,d_loss))
# note that params and d_loss should be in the same shape.

direction = gn_direction(jac_2nd_elliptic)

# functions of numerical error
# note that function <error> takes the current value 
# of params automatically as a hyperparameter. It must
# be used carefully with the update-functions.

error = lambda x: model(params, x) - u_exact(x)
v_error = vmap(error, (0))
v_abs_grad_error = vmap(lambda x: jnp.dot(grad(error)(x), grad(error)(x))**0.5)

def l2_norm(f, quadrature):
    return quadrature(lambda x: f(x)**2)**0.5

# line search subroutine of gauss-newton direction
# grid method (global): rate = (1/2)^m with m \in [N].
# armijo method (local): make sure loss will decay.

grid = jnp.linspace(0, 50, 51)
steps = 0.75**grid
line_search = grid_line_search(loss, steps)

# line_search = armijo_line_search(loss)

# Gauss-Newton training loop, with initialization:
# (1) in normal distribution
# (2) the Xavier's method

for i in range(10):

    seed = 0
    width = 64
    layer_sizes = [5,width,1]
    params = normal_init(layer_sizes, random.PRNGKey(seed))
    
    #num_samples_for_train = 14000 + i*2000
    #training_quadrature = quad_rule.n_rectangle_samples(n_rectangle, num_samples_for_train, K=30)
    #print(num_samples_for_train)

    epochs = 5000
    for epoch in range(epochs):

        d_loss_params = grad(loss)(params)
        gauss_newton = direction(params, d_loss_params)
        params, gn_step = line_search(params, gauss_newton)
    
        # prints
        if epoch % 1 == 0:
            l2_error = l2_norm(v_error, testing_quadrature)
            re_l2_error = l2_error / 1.581
            re_h1_error = (l2_error**2 + l2_norm(v_abs_grad_error, testing_quadrature)**2)**0.5 / 5.213
        
            print(
                f'{epoch} {loss(params)} {re_l2_error} {re_h1_error} {gn_step}'
             )      
