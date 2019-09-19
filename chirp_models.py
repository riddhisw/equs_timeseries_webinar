from __future__ import division, print_function, absolute_import

import numpy as np

def chirp(t, x0, x1, x2, x3):

    """ Return a linear chirp signal s(t) in parameter t where:
    
    x0 - intial phase [dtype: scalar float]
    x1 - linear chirp coefficient in t [dtype: scalar float]
    x2 - quadratic chirp coefficient in t [dtype: scalar float]
    x3 - signal amplitude [dtype: scalar float]
    
    This function supports broadcasting and
    s(t) will be of same dimensions as input t.
        
    """
    
    f_t = x0 + ((x1 * t) + (x2 * t**2))
    
    return x3 * np.sin(f_t)

def geometric_chirp(t, x0, x1, x2, x3):

    """ Return geometric chirp signal s(t) in parameter t where:
    
    x0 - intial phase [dtype: scalar float]
    x1 - linear chirp coefficient in t [dtype: scalar float]
    x2 - geometric factor [dtype: scalar float]
    x3 - signal amplitude [dtype: scalar float]
    
    This function supports broadcasting and
    s(t) will be of same dimensions as input t.
        
    """
    f_t = [x0 + x1*( (x2 ** idx_t -1.0) /  np.log(x2) ) for idx_t in t]
    
    return x3 * np.sin(f_t)

def damped_chirp(t, x0, x1, x2, x3, gamma=0., typechirp='geometric'):

    """ Return a damped chirp signal s(t) in parameter t where:
    
       
    x0 - intial phase [dtype: scalar float]
    x3 - signal amplitude [dtype: scalar float]
    
    x1, x2 - depend on the type of chirp model used - linear or geometric
    
    gamma - amplitude damping coefficient. Default: 0.0 
        [dtype: real, non negative scalar float]
    
    This function supports broadcasting and
    s(t) will be of same dimensions as input t.
        
    """
    if typechirp =="linear":
        return chirp(t, x0, x1, x2, x3) * np.exp( -1.0 * gamma * t)
    
    if typechirp == "geometric":
        return geometric_chirp(t, x0, x1, x2, x3) * np.exp( -1.0 * gamma * t)
