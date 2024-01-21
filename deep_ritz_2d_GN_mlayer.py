import jax.numpy as jnp
import jax.numpy.linalg as alg
import jax.flatten_util
from jax import jacfwd, hessian
from jax import vmap, grad, jit, random

from tool.quadrature import GaussLegendrePiecewise
from tool.model import deep_fc_network
from tool.model import normal_init_mlayer
from tool.gauss_newton import jacobian_matrix, jacobian_matrix_multi_dim
from tool.gauss_newton import gn_direction, grid_line_search_mlayer

########################
# This file may need a
# longer time to compile
########################

# for more precisions
jax.config.update("jax_enable_x64", True)

# activation functions
tanh = lambda x: jnp.tanh(x)
relu2 = lambda x: jnp.where(x>0,x,0)**2
relu3 = lambda x: jnp.where(x>0,x,0)**3

# quadrature for training and testing
quad_rule = GaussLegendrePiecewise(npts=2)
h1 = jnp.array([1/200, 1/200])
h2 = jnp.array([1/300, 1/300])
rectangle = jnp.array([[0.,1.],[0.,1.]])
training_quadrature = quad_rule.rectangle_quadpts(rectangle, h1, K=150)
testing_quadrature = quad_rule.rectangle_quadpts(rectangle, h2, K=100)

# model, exact solution and source term.
# reshape(x,()) returns a single number, not an array object.

model = deep_fc_network(relu3)
u_exact = lambda p: jnp.reshape(jnp.cos(2*jnp.pi*p[...,0:1]) * jnp.cos(2*jnp.pi*p[...,1:2]), ())
rhs = lambda p: jnp.reshape((8*jnp.pi**2+1) * u_exact(p), ())
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
    u_square = lambda p: 0.5 * jnp.reshape(v_model(params, p)**2, (len(p),))
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
        in front of -Delta(u) and u are both set to be 
        constant C=1. This function can be used as a handle
        that takes "params" as input.
    """
    d1_jacobian = jacobian_matrix(d1_model, training_quadrature)
    d2_jacobian = jacobian_matrix_multi_dim(d2_model, training_quadrature)

    return d2_jacobian(params) + d1_jacobian(params)

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

error = lambda p: model(params, p) - u_exact(p)
v_error = vmap(error, (0))
v_abs_grad_error = vmap(lambda p: jnp.dot(grad(error)(p), grad(error)(p))**0.5)

def l2_norm(f, quadrature):
    return quadrature(lambda p: f(p)**2)**0.5

# line search subroutine of gauss-newton direction
# grid method: rate = (1/2)^m with m \in [N].
# armijo method: make sure loss will decay.

grid = jnp.linspace(0, 50, 51)
steps = 0.755**grid
line_search = grid_line_search_mlayer(loss, steps)

# line_search = armijo_line_search(loss)

# Gauss-Newton training loop, with initialization:
# (1) in normal distribution
# (2) the Xavier's method

seed = 0
width = 20
layer_sizes = [2,width,width,1]
params = normal_init_mlayer(layer_sizes, random.PRNGKey(seed)) 

epochs = 2500
for epoch in range(epochs):
    d_loss_params = grad(loss)(params)
    gauss_newton = direction(params, d_loss_params)
    params, gn_step = line_search(params, gauss_newton)
    
    # prints
    if epoch % 1 == 0:
        l2_error = l2_norm(v_error, testing_quadrature)
        h1_error = (l2_error**2 + l2_norm(v_abs_grad_error, testing_quadrature)**2)**0.5
        
        print(
            f'{epoch} {loss(params)} {l2_error} {h1_error} {gn_step}'
        )