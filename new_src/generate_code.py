#order I think is 
#def fock_dm or any other function that creates the density matrix returns dm 
#then add noise to the function def add_state_noise returns rho 
#then after that im not very sure but following the paper we need to calculate the huisimi q functions of the resulting states 
#so use function def Huisimi_ops but inputs are (hilbert_size,betas) not sure where to get them. this returns ops 
#after that def_covert_to_real_ops or def convert_to_complex_ops which wil convert it into something a neural network can take as input
# I think the image does have axis of real and imaginary so I think the code should be something like this 


#generate the 7 different classes
def fock_dm(hilbert_size, n=None):
    """
    Generates a random fock state.
    
    Parameters
    ----------
    n : int
        The fock number
    Returns
    -------
    fock_dm: `qutip.Qobj`
        The density matrix as a quantum object.
    """
    if n == None:
        n = np.random.randint(1, hilbert_size/2 + 1)
    return qutip_fock_dm(hilbert_size, n), -1


dm = fock_dm(4,n=none)
fig,ax = qutip.hinton(dm)
fig.show()


def thermal_dm(hilbert_size, mean_photon_number=None):
    """
    Generates a random thermal state.

    Parameters
    ----------
    mean_photon_number: int
        The mean photon number for the thermal state.

    Returns
    -------
    thermal_dm: `qutip.Qobj`
        The density matrix as a quantum object.
    """
    if mean_photon_number == None:
        mean_photon_number = np.random.uniform(hilbert_size/2)
    return qutip_thermal_dm(hilbert_size, mean_photon_number), -1


def coherent_dm(hilbert_size, alpha=None):
    """
    Generates a random coherent state.

    Parameters
    ----------
    alpha: np.complex
        The displacement parameter. D(alpha)

    Returns
    -------
    rand_coherent: `qutip.Qobj`
        The density matrix as a quantum object.
    """
    if alpha == None:
        alpha = random_alpha(1e-6, 3)
    return qutip_coherent_dm(def add_state_noise(dm, sigma=0.01, sparsity=0.01):
    """
    Adds a random density matrices to the input state.
    
    .. math::
        \rho_{mixed} = \sigma \rho_0 + (1 - \sigma)\rho_{rand}$

    Args:
    ----
        dm (`qutip.Qobj`): Density matrix of the input pure state
        sigma (float): the mixing parameter specifying the pure state probability
        sparsity (float): the sparsity of the random density matrix
    
    Returns:
    -------
        rho (`qutip.Qobj`): the mixed state density matrix
    """
    hilbertsize = dm.shape[0]
    rho  = (1 - sigma)*dm + sigma*(rand_dm(hilbertsize, sparsity))
    rho = rho/rho.tr()
    return rho
hilbert_size, alpha), -1


def gkp(hilbert_size, delta=None, mu = None):
    """Generates a GKP state
    """
    gkp = 0*coherent(hilbert_size, 0)

    c = np.sqrt(np.pi/2)

    if mu is None:
        mu = np.random.randint(2)

    if delta is None:
        delta = np.random.uniform(0.2, .50)

    zrange = range(-20, 20)

    for n1 in zrange:
        for n2 in zrange:        
            a = c*(2*n1 + mu + 1j*n2)
            alpha = coherent(hilbert_size, a)
            gkp += np.exp(-delta**2*np.abs(a)**2)*np.exp(-1j*c**2 * 2*n1 * n2)*alpha

    rho = gkp*gkp.dag()
    return rho.unit(), mu


def binomial(hilbert_size, S=None, N=None, mu=None):
    """
    Binomial code
    """
    if S == None:
        S = np.random.randint(1, 10)
    
    if N == None:
        Nmax = int((hilbert_size)/(S+1)) - 1
        try:
            N = np.random.randint(2, Nmax)
        except:
            N = Nmax

    if mu is None:
        mu = np.random.randint(2)

    c = 1/sqrt(2**(N+1))

    psi = 0*fock(hilbert_size, 0)

    for m in range(N):
        psi += c*((-1)**(mu*m))*np.sqrt(binom(N+1, m))*fock(hilbert_size, (S+1)*m)

    rho = psi*psi.dag()
    return rho.unit(), mu


#adding noise 
def add_state_noise(dm, sigma=0.01, sparsity=0.01):
    """
    Adds a random density matrices to the input state.
    
    .. math::
        \rho_{mixed} = \sigma \rho_0 + (1 - \sigma)\rho_{rand}$

    Args:
    ----
        dm (`qutip.Qobj`): Density matrix of the input pure state
        sigma (float): the mixing parameter specifying the pure state probability
        sparsity (float): the sparsity of the random density matrix
    
    Returns:
    -------
        rho (`qutip.Qobj`): the mixed state density matrix
    """
    hilbertsize = dm.shape[0]
    rho  = (1 - sigma)*dm + sigma*(rand_dm(hilbertsize, sparsity))
    rho = rho/rho.tr()
    return rho

#huisimi q function calc
def husimi_ops(hilbert_size, betas):
    """
    Constructs a list of TensorFlow operators for the Husimi Q function
    measurement at beta values.
    
    Args:
        hilbert_size (int): The hilbert size dimension for the operators
        betas (list/array): N complex values to construct the operator
        
    Returns:
        ops (:class:`tensorflow.Tensor`): A 3D tensor (N, hilbert_size, hilbert_size) of N
                                         operators
    """
    basis = []
    for beta in betas:
        op = qutip_coherent_dm(2*hilbert_size, beta)
        op = Qobj(op[:hilbert_size, :hilbert_size])
        basis.append(op)

    return dm_to_tf(basis)

        cstates += sign * coherent(hilbert_size, -prefactor * alpha * (-((1j) ** mu)))

    rho = cstates * cstates.dag()
    return rho.unit(), mu

    #convert to input data 
    ef convert_to_real_ops(ops):
    """
    Converts a batch of TensorFlow operators to something that a neural network
    can take as input.
    
    Args:
        ops (`tf.Tensor`): a 4D tensor (batch_size, N, hilbert_size, hilbert_size) of N
                           measurement operators

    Returns:
        tf_ops (`tf.Tensor`): a 4D tensor (batch_size, hilbert_size, hilbert_size, 2*N) of N
                           measurement operators converted into real matrices
    """
    tf_ops = tf.transpose(ops, perm=[0, 2, 3, 1])
    tf_ops = tf.concat([tf.math.real(tf_ops), tf.math.imag(tf_ops)], axis=-1)
    return tf_ops


def convert_to_complex_ops(ops):
        cstates += sign * coherent(hilbert_size, -prefactor * alpha * (-((1j) ** mu)))

    rho = cstates * cstates.dag()
    return rho.unit(), mu
    """
    Converts a batch of TensorFlow operators to something that a neural network
    can take as input.
    
    Args:
        ops (`tf.Tensor`): a 4D tensor (batch_size, N, hilbert_size, hilbert_size) of N
                           measurement operators

    Returns:
        tf_ops (`tf.Tensor`): a 4D tensor (batch_size, hilbert_size, hilbert_size, 2*N) of N
                           measurement operators converted into real matrices
    """
    shape = ops.shape
    num_points = shape[-1]
    
    
    tf_ops = tf.complex(ops[..., :int(num_points/2)], ops[..., int(num_points/2):])
    tf_ops = tf.transpose(tf_ops, perm=[0, 3, 1, 2])
    return tf_ops
    
    
 
