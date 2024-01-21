import jax.numpy as jnp
import jax.flatten_util as flatten_util
from jax import jit, vmap, grad, jacfwd
from jax.numpy.linalg import lstsq

def jacobian_matrix(derivative, quadrature):
    
    def jac_single_input(params, x):
        deri = derivative(params, x)
        flat = flatten_util.ravel_pytree(deri)[0]
        flat_col = jnp.reshape(flat, (len(flat), 1))
        flat_row = jnp.reshape(flat, (1, len(flat)))
        return jnp.matmul(flat_col, flat_row)
    
    v_jac = vmap(jac_single_input,(None,(0)))
    
    def jacobian(params):
        jac = quadrature(lambda x: v_jac(params, x))
        return jac

    return jacobian



def jacobian_matrix_multi_dim(derivative, quadrature):
    
    def jac_single_input(params, x):
        deri = derivative(params, x)
        flat = flatten_util.ravel_pytree(deri)[0]
        
        # take flat apart
        jac_single = 0
        arr = jnp.arange(0,len(flat))
        for i in range(len(x)):
            sub_flat = flat[jnp.where((arr%len(x)==i))[0]]
            sub_flat_col = jnp.reshape(sub_flat, (len(sub_flat), 1))
            sub_flat_row = jnp.reshape(sub_flat, (1, len(sub_flat)))
            jac_single += jnp.matmul(sub_flat_col, sub_flat_row)
        return jac_single
    
    v_jac = vmap(jac_single_input,(None,(0)))
    
    def jacobian(params):
        jac = quadrature(lambda x: v_jac(params, x))
        return jac

    return jacobian



def gn_direction(jacobian):
    
    def gauss_newton(params, d_loss_params):
        
        jac_matrix = jacobian(params)
        flat_d_loss_params, retriev_pytree  = flatten_util.ravel_pytree(d_loss_params)
      
        # solve the gauss-newton direction
        flat_gn_direction = lstsq(jac_matrix, flat_d_loss_params)[0]
        
        # if jacobian is zero then lstsq gives back nan...
        if jnp.isnan(flat_gn_direction[0]):
            return retriev_pytree(jnp.zeros_like(flat_gn_direction))
        else:
            return retriev_pytree(flat_gn_direction)
        
    return gauss_newton


def grid_line_search(loss, steps):
    
    def update_params(step, params, d_params):
        w1, b1 = params[0]
        dw1, db1 = d_params[0]
        w2 = params[1]
        dw2 = d_params[1]
        return [(w1 - step * dw1, b1 - step * db1), (w2 - step * dw2)]
    
    def loss_at_step(step, params, d_params):
        return loss(update_params(step, params, d_params))
    
    v_loss_at_step = jit(vmap(loss_at_step, (0, None, None)))
    
    @jit
    def forward(params, d_params):
        losses = v_loss_at_step(steps, params, d_params)
        step = steps[jnp.argmin(losses)]
        return update_params(step, params, d_params), step
    
    return forward


def grid_line_search_mlayer(loss, steps):
    
    def update_params(step, params, d_params):
        updated_params = [(w - step * dw, b - step * db)
            for (w, b), (dw, db) in zip(params, d_params)]
        return updated_params
    
    def loss_at_step(step, params, d_params):
        return loss(update_params(step, params, d_params))
    
    v_loss_at_step = jit(vmap(loss_at_step, (0, None, None)))
    
    @jit
    def forward(params, d_params):
        losses = v_loss_at_step(steps, params, d_params)
        step = steps[jnp.argmin(losses)]
        return update_params(step, params, d_params), step
    
    return forward


def armijo_line_search(loss, c_armijo=1e-12, max_search=100):
    
    def get_loss(params):
        return loss(params)
    
    def get_d_loss_params(params):
        return grad(loss)(params)
    
    def update_params(step, params, d_params):
        w1, b1 = params[0]
        dw1, db1 = d_params[0]
        w2 = params[1]
        dw2 = d_params[1]
        return [(w1 - step * dw1, b1 - step * db1), (w2 - step * dw2)]
    
    def loss_at_step(step, params, d_params):
        return loss(update_params(step, params, d_params))
    
    def forward(params, d_params):
        """
            In this code, d_params is the direction,
            d_loss_params is the gradient of loss,
            step is the learning rate.
        """
        m = 0
        c1 = 1
        c2 = 0.8
        step = c1 * c2 ** m
        
        init_loss = get_loss(params)
        d_loss_params = get_d_loss_params(params)
        flat_direction = flatten_util.ravel_pytree(d_params)[0]
        flat_gradient, retriev_pytree  = flatten_util.ravel_pytree(d_loss_params)
        tgd = step * jnp.dot(flat_direction, flat_gradient)

        # back-tracking method to adjust step
        line_search_iter = 0
        is_armijo_working = 0
        new_loss = loss_at_step(step, params, d_params)
        while line_search_iter <= max_search:
            
            if new_loss <= (init_loss - c_armijo * tgd):
                is_armijo_working = 1
                break
            
            m += 1
            step = c1 * c2 ** m
            new_loss = loss_at_step(step, params, d_params)
            line_search_iter += 1
        
        # Have reached max number of iterations?
        if line_search_iter == max_search:
            step = c1 * c2 ** m
            
        return update_params(step, params, d_params), step, is_armijo_working
       
    return forward










