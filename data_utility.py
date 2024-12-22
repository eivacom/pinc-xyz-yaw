import numpy as np

def random_input(t, N_u, input_type='noise'):
    """
    Generates random input signals for the system.
    
    Args:
        t (ndarray): Time array of shape (N,).
        N_u (int): Number of control inputs.
        input_type (str): Type of input signal ('noise', 'sine', 'noise_x', 'sine_x').
        
    Returns:
        U (ndarray): Input signal array of shape (N, N_u).
    """
    N = len(t)
    U = np.zeros((N, N_u))
    
    if input_type in ['noise', 'noise_x']:
        # Generate noise input
        u_signs = np.random.choice([-1, 1], size=N_u)
        u_offsets = np.random.normal(0, 0.5, size=N_u)
        m_indices = np.arange(N)
        half_N = N // 2
        
        # Ramp up and down profiles
        ramp_up = (u_signs * m_indices[:half_N, None] / half_N - 0.5 * u_signs + u_offsets)
        ramp_down = (u_signs * (N - m_indices[half_N:, None]) / half_N - 0.5 * u_signs + u_offsets)
        U_noise = np.vstack((ramp_up, ramp_down))
        
        # Alternative code with random pulses like pseudo random trajectories. Can be used in future work for more exciting signals
        '''for n in range(N_u):
            u_sign = np.random.choice([-1, 1])
            for m in range(N):
                if m%4==0:
                    u = np.random.normal(0, 0.5, 1)
                if m < 0.5*N:
                    U_noise[m, n] = u_sign*m/(0.5*N) - 0.5*u_sign + u
                else:
                    U_noise[m, n] = u_sign*(N-m)/(0.5*N) - 0.5*u_sign + u'''
        
        if input_type == 'noise':
            U += U_noise
        elif input_type == 'noise_x':
            U[:, 0] += U_noise[:, 0]
    
    elif input_type in ['sine', 'sine_x']:
        # Generate sine input
        sine_freq = np.random.uniform(0.01, 0.2, N_u)
        sine_phase = np.random.uniform(0, 2 * np.pi, N_u)
        sine_amp = 3  # Scalar amplitude
        U_sine = sine_amp * np.sin(2 * np.pi * sine_freq * t[:, None] + sine_phase)
        
        if input_type == 'sine':
            U += U_sine
        elif input_type == 'sine_x':
            U[:, 0] += U_sine[:, 0]
    
    else:
        raise ValueError(f"Invalid input_type '{input_type}'. Must be 'noise', 'sine', 'noise_x', or 'sine_x'.")
    
    return U

def random_x0(intervals):
    """
    Generates a random initial state vector x_0 based on specified intervals.
    
    Args:
        intervals (list or ndarray): Intervals for each state variable. Should have length 8.
        
    Returns:
        x_0 (ndarray): Initial state vector of shape (8,).
    """
    intervals = np.asarray(intervals)
    if intervals.shape[0] != 8:
        raise ValueError("intervals must have length 8.")
    
    # Generate random positions
    p = np.random.uniform(-intervals[0:3], intervals[0:3])
    
    # Generate random heading angle psi
    psi = np.random.uniform(-intervals[3], intervals[3])
    
    # Generate random velocities
    v = np.random.uniform(-intervals[4:7], intervals[4:7])
    v[2] = np.abs(v[2])  # Only dive (positive vertical velocity)
    
    # Generate random angular velocity w_r
    w_r = np.random.uniform(-intervals[7], intervals[7])
    
    # Concatenate all state variables
    x_0 = np.concatenate((p, [psi], v, [w_r]))
    
    return x_0
