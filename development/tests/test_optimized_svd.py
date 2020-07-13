import numpy as np

from development.print_functions.svd_print_machine import svd_optimized_printer

if __name__ == '__main__':
    matrix = np.array([[1, 2], [3, 4],
                       [5, 6], [7, 8],
                       [9, 10], [11, 12]])

    svd_optimized_printer(matrix)
