from numpy import random, matmul, trace, array, sum, kron
from numpy import eye, transpose
from scipy.linalg import eigvals


def generate_hermitian_product_states(size, n_matrices):
    """Generates a list of random Hermitian product states
    of the given dimension
    Product states are hermitian matrices with trace 1.
    Input:
        size: size of the matrices
        n_matrices: number of matrices to generate
    Output:
        product_states: 3D numpy array of product states
    """

    product_states = []
    for _ in range(n_matrices):
        real_part = random.rand(size, size)
        imag_part = random.rand(size, size)
        product_state = real_part + 1j * imag_part
        product_state = matmul(product_state, product_state.conj().T)
        product_state /= trace(product_state)
        product_states.append(product_state)

    return array(product_states)


def generate_coefficients(n):
    """
    Generates a list of n random coefficients that sum to 1
    Input:
        n: number of coefficients to generate
    Output:
        coefficients: numpy array of n coefficients
    """
    rand_numbers = random.rand(n)
    rand_numbers /= sum(rand_numbers)

    return rand_numbers


def generate_separable_states(n_matrices, n_states):
    """Generates a list of random separable states of the given dimension
    Input:
        dimensions: size of the matrices
        n_matrices: number of matrices used to generate the states
        n_states: number of separable states to generate
    Output:
        separable_states: 3D numpy array of separable states,
        of size n_states x dimensions^n_matrices
    """

    states = []

    for _ in range(n_states):
        rhoA = generate_hermitian_product_states(2, n_matrices)
        rhoB = generate_hermitian_product_states(2, n_matrices)
        coeffs = generate_coefficients(n_matrices)

        sep_state = 0

        for i in range(n_matrices):
            sep_state += coeffs[i] * kron(rhoA[i], rhoB[i])

        states.append(sep_state)

    return array(states)


# Entanglement check
def is_entangled(rho):
    # Check if the density matrix is 4x4
    if rho.shape != (4, 4):
        raise ValueError("The input matrix should be a 4x4 density matrix.")

    # Calculate the partial transpose of the density matrix
    pauli_mat_B = array([[1, 0], [0, -1]])
    identity_mat = eye(2)
    
    transpose_op = kron(identity_mat, pauli_mat_B)
    rho_T_B = transpose_op @ transpose(rho) @ transpose_op

    # rho_T_B = kron(eye(2), array([[1, 0], [0, -1]])) @
    # conjugate(transpose(rho)) @
    # kron(eye(2), array([[1, 0], [0, -1]]))

    # Check if any eigenvalue of the partial transpose is negative
    eigenvalues = eigvals(rho_T_B)

    return any(eig < 0 for eig in eigenvalues)


def generate_entangled_states(n_states):

    states = []
    i = 0
    while i < n_states:
        rand_state = random.rand(4, 4)

        if is_entangled(rand_state):
            states.append(rand_state)
            i += 1
        else:
            continue
    return array(states)
