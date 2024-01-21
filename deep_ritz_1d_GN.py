import jax.numpy as jnp
import jax.numpy.linalg as alg
import jax.flatten_util
from jax import jacfwd, hessian
from jax import vmap, grad, jit, random

from tool.model import shallow_network
from tool.model import normal_init
from tool.quadrature import GaussLegendrePiecewise
from tool.gauss_newton import jacobian_matrix, gn_direction
from tool.gauss_newton import armijo_line_search, grid_line_search

# for more precisions
jax.config.update("jax_enable_x64", True)

# activation functions
tanh = lambda x: jnp.tanh(x)
relu2 = lambda x: jnp.where(x>0,x,0)**2
relu3 = lambda x: jnp.where(x>0,x,0)**3

# quadrature for training and testing
quad_rule = GaussLegendrePiecewise(npts=2)
h1 = jnp.array([1/3000])
h2 = jnp.array([1/4000])
interval = jnp.array([[-1.,1.]])
training_quadrature = quad_rule.interval_quadpts(interval, h1)
testing_quadrature = quad_rule.interval_quadpts(interval, h2)

# model, exact solution and source term.
# reshape(x,()) returns a single number, not an array object.

model = shallow_network(relu3)
u_exact = lambda x: jnp.reshape(jnp.cos(jnp.pi * x), ())
rhs = lambda x: (1+jnp.pi**2) * u_exact(x)
v_model = vmap(model, (None, (0)))
v_u_exact = vmap(u_exact, (0))
v_rhs = vmap(rhs, (0))

# loss function of deep ritz methods
# for the equation -Delta(u) + u = f
# with Neumann's boundary du/dn = 0.

@jit
def loss_gradient(params):
    nabla_u = vmap(grad(lambda x: model(params, x)), (0))
    nabla_u_square = lambda x: 0.5 * jnp.reshape(nabla_u(x)**2, (len(x),))
    return training_quadrature(nabla_u_square)

@jit
def loss_linear_term(params):
    u_square = lambda x: 0.5 * jnp.reshape(v_model(params, x)**2, (len(x),))
    return training_quadrature(u_square)

@jit 
def loss_rhs(params):
    u_dot_f = lambda x: jnp.reshape(v_model(params, x)*v_rhs(x), (len(x),))
    return training_quadrature(u_dot_f)

@jit
def loss(params):
    return loss_gradient(params) + loss_linear_term(params) - loss_rhs(params)

# two types of jacobian concerning the
# loss function of -Delta(u) + u = f.

def dt_di_model(model):
    
    def dt_model(params, x):
        return grad(model, (0))(params, x)
    
    def dtdi_single_input(params, x):
        return jacfwd(dt_model, (1))(params, x)
    
    return dtdi_single_input  

def dt_model(model):
    
    def dt_single_input(params, x):
        return grad(model, (0))(params, x)
    
    return dt_single_input  


d1_model = dt_model(model)
d2_model = dt_di_model(model)
    
def jac_2nd_elliptic(params):
    """
        The Jacobian matrix for the escond order 
        elliptic equation. In this case, the coefficients 
        in front of -Delta(u) and u are both set to have 
        constant C=1. This function can be used as a handle
        that takes "params" as input.
    """
    d2_jacobian = jacobian_matrix(d2_model, training_quadrature)
    d1_jacobian = jacobian_matrix(d1_model, training_quadrature)

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

error = lambda x: model(params, x) - u_exact(x)
v_error = vmap(error, (0))
v_abs_grad_error = vmap(lambda x: jnp.dot(grad(error)(x), grad(error)(x))**0.5)

def l2_norm(f, quadrature):
    return quadrature(lambda x: f(x)**2)**0.5

# line search subroutine of gauss-newton direction
# grid method: rate = (1/2)^m with m \in [N].
# armijo method: make sure loss will decay.

grid = jnp.linspace(0, 40, 41)
steps = 0.3**grid
line_search = grid_line_search(loss, steps)

# line_search = armijo_line_search(loss)

# Gauss-Newton training loop, with initialization:
# (1) in normal distribution
# (2) the Xavier's method

seed = 0
width = 2**6
layer_sizes = [1,width,1]
params = normal_init(layer_sizes, random.PRNGKey(seed))

epochs = 1000
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
        
        