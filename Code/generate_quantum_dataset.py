from numpy import random, matmul, trace, array, sum, zeros, complex128, kron


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


def generate_coefficients(n, entangled):
    """
    Generates a list of n random coefficients that sum to 1
    Input:
        n: number of coefficients to generate
    Output:
        coefficients: numpy array of n coefficients
    """
    rand_numbers = random.rand(n)

    if not entangled:
        rand_numbers /= sum(rand_numbers)

    return rand_numbers


def generate_states(dimensions, n_matrices, n_states, entangled):
    """Generates a list of random separable states of the given dimension
    Input:
        dimensions: size of the matrices
        n_matrices: number of matrices used to generate the states
        n_states: number of separable states to generate
    Output:
        separable_states: 3D numpy array of separable states,
        of size n_states * dimensions^n_matrices
    """
    states = []

    for _ in range(n_states):
        product_states = generate_hermitian_product_states(dimensions,
                                                           n_matrices)

        coeffs = generate_coefficients(n=n_matrices, entangled=entangled)

        state = zeros(dimensions ** n_matrices, dtype=complex128)

        for j in range(dimensions):
            result = product_states[0][:, j]

            for i in range(1, n_matrices):
                result = kron(result, product_states[i][:, j])
            result *= coeffs[j]

            state += result

        states.append(state)

    return array(states)
