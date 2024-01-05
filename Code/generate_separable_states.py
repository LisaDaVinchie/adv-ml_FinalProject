import numpy as np


def generate_hermitian_product_states(dimensions):

    product_states = []
    for dim in dimensions:
        real_part = np.random.rand(dim[0], dim[1])
        imag_part = np.random.rand(dim[0], dim[1])
        product_state = real_part + 1j * imag_part
        product_state = np.matmul(product_state, product_state.conj().T)
        product_state /= np.trace(product_state)
        product_states.append(product_state)

    return np.array(product_states)


def generate_coefficients(n):

    rand_numbers = np.random.rand(n)
    numbers_sum = np.sum(rand_numbers)

    return rand_numbers / numbers_sum


def calculate_tensor_product(herm_matrices, coeffs):

    tensor_dot_product = []
    for i in range(len(coeffs)):
        temp_product = np.tensordot(herm_matrices[0, :, i],
                                    herm_matrices[1, :, i],
                                    axes=0)

        if herm_matrices.shape[0] > 2:
            for j in np.arange(2, herm_matrices.shape[0]):
                temp_product = np.tensordot(temp_product,
                                            herm_matrices[j, :, i],
                                            axes=0)

    tensor_dot_product.append(coeffs[i] * temp_product)

    return np.add.reduce(tensor_dot_product)


def generate_separable_states(n_rows, n_qubits, n_states):

    sep_states = []
    for _ in range(n_states):
        herm_matrices = generate_hermitian_product_states(n_rows, n_qubits)
        coeffs = generate_coefficients(n_rows)
        tensor_dot_product = calculate_tensor_product(herm_matrices, coeffs)
        sep_states.append(tensor_dot_product / np.linalg.norm(tensor_dot_product))
    return np.array(sep_states)
