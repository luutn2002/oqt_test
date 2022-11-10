#order I think is 
#def fock_dm or any other function that creates the density matrix returns dm 
#then add noise to the function def add_state_noise returns rho 
#then after that im not very sure but following the paper we need to calculate the huisimi q functions of the resulting states 
#so use function def Huisimi_ops but inputs are (hilbert_size,betas) not sure where to get them. this returns ops 
#after that def_covert_to_real_ops or def convert_to_complex_ops which wil convert it into something a neural network can take as input
# I think the image does have axis of real and imaginary so I think the code should be something like this 


import qutip

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



