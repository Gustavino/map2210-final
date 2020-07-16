import numpy as np

from development.print_functions.svd_print_machine import svd_optimized_printer

if __name__ == '__main__':
    # matrix = np.array([[1, 2, 3],
    #                    [4, 5, 6],
    #                    [7, 8, 9],
    #                    [10, 11, 12]])

    matrix = np.array([[1, 2], [3, 4],
                       [5, 6], [7, 8],
                       [9, 10], [11, 12]])

    svd_optimized_printer(matrix)

    # The rows of `vh` are the eigenvectors of :math:`A^H A` and the columns of `u` are
    # the eigenvectors of :math:`A A^H`.

    MMT = matrix @ np.transpose(matrix)
    MTM = np.transpose(matrix) @ matrix

    # print("MMT: \n {}".format(MMT))
    # print("MTM: \n {}".format(MTM))

    mmt_eigvalues, mmt_eigvectors = np.linalg.eig(MMT)
    mtm_eigvalues, mtm_eigvectors = np.linalg.eig(MTM)

    mmt_eigvalues = np.array(mmt_eigvalues).real
    mtm_eigvalues = np.array(mtm_eigvalues).real

    mmt_eigvectors = np.array(mmt_eigvectors).real
    mtm_eigvectors = np.array(mtm_eigvectors).real

    # mmt_eigvectors[:, 1] *= -1

    mmt_eigvalues = np.sqrt(mmt_eigvalues)
    mtm_eigvalues = np.sqrt(mtm_eigvalues)

    print("MMT Eigenvectors:\n {}".format(mmt_eigvectors))
    print("MTM Eigenvectors: \n {}".format(mtm_eigvectors))

    sigma = np.zeros((matrix.shape[0], matrix.shape[1]))
    sigma[:matrix.shape[1], :matrix.shape[1]] = np.diag(mtm_eigvalues)

    print("MTM Eigenvalues: \n {}".format(sigma))

    # print("MMT Eigenvalues: \n {}".format(mmt_eigvalues))
    # mmt_values[2] = 10**-18

    print("Original matrix: \n {}".format(mmt_eigvectors @ sigma @ np.transpose(mtm_eigvectors)))
