# David Omrai 23.12.2021
import numpy as np

# Maticove operace ----------------------------------------------------------------
"""
    Soucet matic, na vstup prijdou dve matice,
    ktere se nejprve zkontroluji zdali je lze 
    secist a v pozitivnim pripade toto probehne
    za pomoci pretizeneho operatoru + knihovny numpy
"""
def sum_matrices(matrix1, matrix2):
    if (matrix1.shape != matrix2.shape):
        return Exception("Matrices has different shape")
    return  matrix1 + matrix2

"""
    Odecitani matic, stejne jako u scitani matic,
    jen s jinym pretizenym operatorem
"""
def sub_matrixes(matrix1, matrix2):
    if (matrix1.shape != matrix2.shape):
        return Exception("Matrices has different shape")
    return matrix1 - matrix2

"""
    Nasobeni matic, na vstup opet prijdou dve 
    matice, ktere se po kontrole za pomoci metody
    z knihovny numpy vynasobi
"""
def mult_matrices(matrix1, matrix2):
    if (matrix1.shape[1] != matrix2.shape[0]):
        return Exception("Matrices can't be multiplied")
    return np.matmul(matrix1, matrix2)

"""
    Inversni matice, vstupni matice se zkontroluje zdali 
    neni singularni a pokud neni, tak se za pomoci metody
    z knihovny numpy spocte jeji inverze
"""
def inverse_matrix(matrix):
    m_det = np.linalg.det(matrix)

    # zkontroluj zdali k matici existuje inverzni
    if (m_det == 0):
        return Exception("No inverse matrix possible, the matrix is singular")
    return np.linalg.inv(matrix)

# Rozhlad matice A ----------------------------------------------------------------

"""
    Metoda vraci L matici, kde se nachazi
    prvky pod diagonalou, ostatni jsou nuly
"""
def get_l_matrix(a_matrix):
    l_matrix = np.copy(a_matrix)
    for i in range(0, l_matrix.shape[0]):
        for j in range(i, l_matrix.shape[1]):
            l_matrix[i][j] = 0
    return l_matrix

"""
    Metoda vraci U matici, kde se zachovavaji
    prvky nad diagonalou, ostatni jsou nuly
"""
def get_u_matrix(a_matrix):
    u_matrix = np.copy(a_matrix)
    for i in range(0, u_matrix.shape[0]):
        for j in range(i, u_matrix.shape[1]):
            u_matrix[j][i] = 0
    return u_matrix

"""
    Metoda vraci D matici, kde se zachovavaji prvky
    pouze na diagonale, ostatni jsou nuly
    Metoda vyuziva U a L matici k vypoctu
"""
def get_d_matrix(a_matrix):
    l_matrix = get_l_matrix(a_matrix)
    u_matrix = get_u_matrix(a_matrix)

    return a_matrix - l_matrix - u_matrix

# Priprava dat pro ulohu ----------------------------------------------------------

def get_a_matrix(y):
    a_matrix = np.zeros((20, 20), dtype=float)

    a_matrix[0, 0] = y
    for i in range(1, a_matrix.shape[0]):
        a_matrix[i, i] = y
        a_matrix[i, i-1] = -1.
        a_matrix[i-1, i] = -1.

    return a_matrix

def get_b_vector(y):
    b_vector = np.zeros((20, 1), dtype=float)

    b_vector[0, 0] = y-1
    b_vector[19, 0] = y-1
    for i in range(1, b_vector.shape[0]-1):
        b_vector[i, 0] = y-2
    
    return b_vector

def is_criteria_satisfied(a_matrix, b_vector, x_vector):
    return (
        np.linalg.norm(
            mult_matrices(a_matrix, x_vector) - b_vector)/
        np.linalg.norm(b_vector)) < 10**-6
    

# Iteracni metody -----------------------------------------------------------------
def jacobi_method():
    """todo"""

def sor_method():
    """todo"""


x = np.zeros((20,1))

print(is_criteria_satisfied(get_a_matrix(5), get_b_vector(5), x))