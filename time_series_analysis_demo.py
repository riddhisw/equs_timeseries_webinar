from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.misc as spy

# ===========================================================================
# ===========================================================================

## General Kalman Filtering:


def kalman_filter(y_t, dims=1, ydims=1,
                  x0=1., p0=100000.,
                  Q_var=None, R_var=None, 
                  f_matrix=None, h_matrix=None,
                  t_steps_ahead=10,
                  non_lin_msmt_func=False,
                  state_dependent_h=False,
                  model_type=None):
    
    '''
    y_t: 
        Measurement data (observed signal) indexed by t = 1, 2, ... num.
        Type: (num, m_dims) numpy array
    dims: 
        Dimensions of state space for x.
        Type: int scalar
        Default value: 1 
    x0: 
        Kalman initial condition for x at t=0
        Type: float scalar
        Default value: 1 
    p0: 
        Kalman initial condition for P at t=0
        Type: float scalar
        Default value: 100000
    Q_var: 
        Kalman process noise covariance matrix
        Type: (dims, dims) numpy array
        Default value: None    
    R_var: 
        Kalman measurement noise covariance matrix
        Type: (m_dims, m_dims) numpy array 
        Default value: None 
    f_matrix: 
        Kalman matrix representation of dynamical model 
        Type: 
        Default value: None 
    h_matrix: 
        Kalman matrix representation of measurement model
        Type: (m_dims, dims) numpy array
        Default value: None
    t_steps_ahead: 
        Number of t-steps for which a prediction is desired once data-signal y_t ends. 
        Type: int scalar
        Default value: 10 
    non_lin_msmt_func:
        Kalman h(.) if h is non linear. 
        If True, calc_non_lin_H must be user defined.
        Type: Boolean (True, False)
        Default value: False
    state_dependent_h: 
        True / False flag if h(.) is state dependent.
        If True, calc_state_depend_H must be user defined.
        Type: Boolean (True, False)
        Default value: False
    model_type:
        Type: "None", "chirp", or "sinsuoid"
        Default value: None      
        Selects the correct non_lin_msmt_func h(dot)
        and the correct (state dependent) matrix representation.
        Corresponds to Section 2.1 and 3.4 in webinar materials.
    '''
    
    num =  len(y_t) 
    
    # Set model here
    if Q_var is None:
        Q_var = np.identity(dims)
    if R_var is None:
        R_var = np.identity(1.)
    if f_matrix is None:
        f_matrix = np.identity(dims)
    if h_matrix is None:
        h_matrix = np.zeros((1, dims)) 
    

    # Store estimates
    x_hat = np.zeros((num + t_steps_ahead, dims, 1))
    P_hat = np.zeros((num + t_steps_ahead, dims, dims))
    gains = np.zeros((num + t_steps_ahead, dims, ydims))
    errors = np.zeros((num + t_steps_ahead, ydims))
    
    # Set initial condition
    if type(x0) == float:
        x_hat[0, :, 0] = x0 * np.ones(dims)
    if type(x0) != float and len(x0) == dims:
        x_hat[0, :, 0] = x0
    P_hat[0, :, :] = p0 * np.identity(dims)
    
    if state_dependent_h is True:
        h_matrix = calc_state_depend_H(x_hat[0, :, 0], model_type=model_type)
    
    # Run filter
    idx_t = 1
    while (idx_t < num + t_steps_ahead):
        
        msmt_= None
        if idx_t < num:
            msmt_= y_t[idx_t]
        
        output  = kalman_step(idx_t, x_hat[idx_t-1, :, 0],
                              P_hat[idx_t-1, :, :], 
                              f_matrix, 
                              h_matrix, 
                              Q_var, 
                              R_var,
                              msmt=msmt_,
                              non_lin_msmt_func=non_lin_msmt_func,
                              state_dependent_h=state_dependent_h,
                              model_type=model_type)
        
        x_hat[idx_t, :, 0] = output[0]
        P_hat[idx_t, :, :] = output[1]
        gains[idx_t] = output[2]
        
        if output[3] is not None:
            errors[idx_t] = output[3].flatten() 
        
        idx_t = idx_t + 1
    
    return x_hat, P_hat, gains, errors


def kalman_step(t_index, x_hat, P_hat, 
                f_matrix, h_matrix, Q_var, R_var,
                msmt=None,
                non_lin_msmt_func=False,
                state_dependent_h=False,
                model_type=None): 
    
    '''Return posterior (x, P) for after recieving a single msmt at t. '''
    
    
    # Evolve state estimates from t-1 to t
    x_hat_apriori, P_hat_apriori = propagate_states(f_matrix, x_hat, P_hat, Q_var)
    
    # Check if prediction is required
    if msmt is None:
        return x_hat_apriori, P_hat_apriori, 0.0, None # equivalent to setting gain to zero
    
    # Accomodate state dependent h
    if state_dependent_h is True:
        h_matrix = calc_state_depend_H(x_hat_apriori,
                                       model_type=model_type)
    
    # Compute Kalman gain 
    gain, helper_S = calc_kalman_gain(h_matrix, P_hat_apriori, R_var)
        
    # Compute residual between actual and predicted msmt
    residual = calc_residuals(h_matrix, x_hat_apriori, msmt, non_lin_msmt_func, model_type=model_type)
    
    # Update state estimates
    x_hat, P_hat = estimate_posterior(x_hat_apriori, P_hat_apriori, gain, helper_S, residual)
    
    return [x_hat, P_hat, gain, residual]


def propagate_states(f_matrix, x_hat, P_hat, Q_var):
    '''Return propagated (x, P) using the Kalman dynamic model.

    Parameters:
    ----------
        f_matrix (`float64`): Kalman dynamical model for evolution of x_hat.
        x_hat (`float64`): Kalman state estimate (posterior at previous time step).
        P_hat (`float64`): Kalman state covariance estimate (posterior at previous time step).
        Q (`float64`): Kalman process noise variance scale.

    Returns:
    -------
        x_hat_apriori : Kalman state vector (prior at current time step).
        P_hat_apriori : Kalman state covariance matrix (prior at current time step).
        Q : Kalman process noise covariance matrix (due to white noise input
            at current time step).
    '''
    x_hat_apriori = np.dot(f_matrix, x_hat)
    P_hat_apriori = np.dot(np.dot(f_matrix, P_hat), f_matrix.T) + Q_var

    return x_hat_apriori, P_hat_apriori


def calc_kalman_gain(h_matrix, P_hat_apriori, R_var):
    '''Return the Kalman gain.

    Parameters:
    ----------
        h_matrix (`float64`) : Kalman measurement model.
        P_hat_apriori (`float64`) : Kalman state variance matrix.
        R_var (`float64`) : Kalman measurement noise variance scale.

    Returns:
    -------
        W_gain : Kalman gain / Bayesian update for Kalman state estimates.
        S : Intermediary covariance matrix for calculating Kalman gain.
            NB: S can be a matrix iff R_var is a matrix, and S_inv = np.linalg.inverse(S)
            instead of 1.0/S.
    '''        
    S = np.dot(h_matrix, np.dot(P_hat_apriori, h_matrix.T)) + R_var
    S_inv = np.linalg.inv(S) # 1.0/S and np.linalg.inv(S) are equivalent when S is rank 1

    if not np.isfinite(S_inv).all():
        # Add checks for positive definite and symmetric S
        print("S is not finite")
        raise RuntimeError

    W_gain = np.dot(np.dot(P_hat_apriori, h_matrix.T), S_inv)
    
    return W_gain, S 

def calc_residuals(h_matrix, x_hat_apriori, msmt, non_lin_msmt, model_type=None):
    '''Return residuals between one step ahead predictions and measurement data.

    Parameters:
    ----------
        h_matrix ('float64') : Kalman measurement model.
        x_hat_apriori ('float64') : Kalman state vector.
        msmt ('float64') : Input measurement data:
        non_lin_msmt : Default None. If not None, must supply the function h(x)
    '''
    residual = np.zeros((h_matrix.shape[0], 1))
    if non_lin_msmt is False:
        residual[:, 0] =  msmt - np.dot(h_matrix, x_hat_apriori)
    
    if non_lin_msmt is True:
        residual[:, 0] =  msmt - calc_non_lin_H(x_hat_apriori, model_type=model_type) 
    return residual

def estimate_posterior(x_hat_apriori, P_hat_apriori, W_gain, S, residual):
    '''Return posterior (x, P) using the Kalman gain and msmt residual.
    
    Parameters:
    ----------
        x_hat_apriori : Kalman state estimate before incoming msmt.
        P_hat_apriori : Kalman state uncertainty estimate before incoming msmt.
        W_gain : Kalman gain.
        S : Intermdiary Kalman covariance matrix 
        residual : Error between projected one-step ahead msmt and actual msmt. 
        
    '''
    x_hat = x_hat_apriori + np.dot(W_gain, residual).flatten()
    P_hat = P_hat_apriori - np.dot(np.dot(W_gain, S), W_gain.T) 
    
    return x_hat, P_hat


# ===========================================================================
# ===========================================================================

# Application Specific Models of Section 2 and 3

def calc_state_depend_H(x_state, model_type=None):
    
    dims = len(x_state.flatten())
    if model_type == "sinusoid":
        phi, omega, A = x_state

        dims = len(x_state)
        h_matrix = np.zeros((1, dims))
        h_matrix[0,0] = A * np.cos(phi)
        h_matrix[0,1] = 0.
        h_matrix[0,2] = np.sin(phi)
        return h_matrix
    
    if model_type == "chirp":
        h_matrix = np.zeros((2, dims))
        h_matrix[:, 0] = [np.cos(x_state[2]), np.sin(x_state[2])]
        h_matrix[:, 2] = [-1.*x_state[0]*np.sin(x_state[2]), x_state[0]*np.cos(x_state[2])]
        return h_matrix
    
    print("No state dependent model specified")
    raise RuntimeError
    

def calc_non_lin_H(x_state, model_type=None):
    
    if model_type == "sinusoid":
        phi, omega, A = x_state
        return A*np.sin(phi)
    
    if model_type == "chirp":
        
        result = np.zeros(2)
        result[0] = x_state[0] * np.cos(x_state[2])
        result[1] = x_state[0] * np.sin(x_state[2])
        
        return result

    print("No non-linear model specified")
    raise RuntimeError
        

def calc_chirp_dynamics(phase_order_M, delta_t):
    
    phi_row_vector =  [1./ float(spy.factorial(idx, exact=True)) for idx in range(phase_order_M + 1)]
    phi_matrix  = np.zeros((phase_order_M+1, phase_order_M+1))
    f_matrix = np.eye(phase_order_M + 3)
    
    for idx in range(0, phase_order_M + 1, 1):
        phi_matrix[idx, idx :] = phi_row_vector[0: phase_order_M + 1 - idx]

    f_matrix[2:, 2:] = phi_matrix
    f_matrix[0, 1]= -1. * delta_t
    
    return f_matrix

def psd_ar(order, f_vals, phi_vector, true_Q_var):
    '''
    Return S(f) / Delta_t in units of signal^2.
    
    Let Delta_t be time domain spacing in seconds
    Typically one computes S(f) in units of signal^2/Hz == signal^2 s
    But here, S(f) / Delta_t only inherits units from signal variance
    
    Next, note f_vals are some set of points between [0, 1].
    f_vals represent the ratio of |f| to the sampling rate f_sampling.
    Let f_sampling = 1 / Delta_t
    Then |f| <= f_sampling 
    This implies 0 <= |f| / f_sampling <= 1
    
    '''
    
    if order == 0:
        return true_Q_var

    if order >= 0:
        
        product= np.zeros_like(f_vals, dtype='complex128')
        
        for idx_k in range(1, order + 1, 1):
            product += phi_vector[idx_k-1] * np.exp(-1j*2.*np.pi*f_vals*idx_k)

        return true_Q_var / np.abs(1. - product)**2

def get_autoreg_model(order, weights):
    """ Return the dynamic state space model for AR(q) process.

    Parameters:
    ----------
        order (`int`) : order q of an AR(q) process [Dim:  1x1].
        weights :  coefficients of an AR(q) process [Dim: 1 x order].

    Returns:
    -------
        a (`float64`):  state space dynamics for AR(q) process in AKF [Dim: order x order].
    """
    a = np.zeros((order, order))

    # Allocate weights
    a[0, :] = weights
    # Pick off past values
    idx = range(order-1)
    idx2 = range(1, order, 1)
    a[idx2, idx] = 1.0

    return a

# ===========================================================================
# ===========================================================================

# Test Function for Section 2

def test_function(t, name, msmt_noise_var, 
                  delta_t=1.,
                  multi_signal_starts=[0]):
    
    num = len(t)
    delta_f = 1./ (delta_t * num)
    
    signal = np.zeros(num)
    scale_factor = 1./float(len(multi_signal_starts))
    amp = 1.
    
    components=[]
    if name== "sinusoid" or name== "liska":
                
        for start_time in multi_signal_starts:
            
            phs = np.random.uniform() * np.pi
            
            freq = np.random.uniform(low=delta_f*2., high=delta_f*10.)     
            components.append(freq)#print("Freq Component: %s Hz" %(np.round(freq,3)))   

            signal[start_time :] += amp * scale_factor * np.sin(2.*np.pi*freq*t[start_time :] + phs) 
        
    noise = amp*np.random.normal(loc=0., scale=np.sqrt(msmt_noise_var), size=len(t))

    return signal,  signal + noise, components
    
