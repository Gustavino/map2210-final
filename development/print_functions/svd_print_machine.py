import numpy as np


def svd_optimized_printer(matrix):
    __svd_printer__(matrix, True)


def svd_full_printer(matrix):
    __svd_printer__(matrix, False)


def __svd_printer__(matrix, is_optimized):
    U, S, VT = np.linalg.svd(matrix, full_matrices=is_optimized)

    Sigma = np.zeros((matrix.shape[0], matrix.shape[1]))
    Sigma[:matrix.shape[1], :matrix.shape[1]] = np.diag(S)

    print("U:\n {}".format(U))

    print("VT transposed: \n {}".format(np.transpose(VT)))

    print("Sigma: \n {}".format(Sigma))

    if is_optimized:
        print("Original matrix: \n {}".format(U @ Sigma @ VT))
    else:
        print("Original matrix: \n {}".format(U @ np.diag(S) @ VT))


