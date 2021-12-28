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

def get_x0_vector():
    return np.zeros((20,1), dtype=float)


def is_criteria_satisfied(a_matrix, b_vector, x_vector):
    return (
        np.linalg.norm(
            mult_matrices(a_matrix, x_vector) - b_vector)/
        np.linalg.norm(b_vector)) < 10**-6
    
"""
    Funkce testuej zdali je matice diagonalne dominantni
    tuto funkci vyuziva Jacobiho a GS metodou k urceni zda-li 
    bude k reseni konvergovat ci o tom nelze rozhodnout
"""
def is_diagonal_dominant(matrix):
    # Otestovani spravnych rozmeru
    if (matrix.shape[0] != matrix.shape[1]):
        return False
    # Otestovani diagonalni dominance sloupcu a radku
    row_dom = True
    col_dom = True
    for i in range(0, matrix.shape[0]):
        row_sum = 0
        col_sum = 0
        for j in range(0, matrix.shape[1]):
            if i != j:
                row_sum += abs(matrix[i, j])
                col_sum += abs(matrix[j, i])
        
        if abs(matrix[i, i]) < row_sum:
            row_dom = False
        if abs(matrix[i, i]) < col_sum:
            col_dom = False
        if(row_dom == False and col_dom == False):
            return False
    return True

"""
    Funkce testuje zdali je matice pozitivne definitni
    a to pomoci jejich vlastnich cisel, neb kazda matice
    symetricka, jejiz vlastni cisla jsou kladna je pozitivne
    definitni

    Funkce je vyuzita pro GS metodu, kde by jeji splneni znamenalo
    ze metoda konverguje k reseni soustavy, v opacnem pripade by
    se o konvergenci nedalo rozhodnout
"""
def is_positive_definite(matrix):
    # Otestovani symetricnosti matice
    if not np.all( matrix - matrix.transpose() == 0):
        return False

    # Otestovani pozitivni difinitnosti
    return np.all(np.linalg.eigvals(matrix) > 0)

# Iteracni metody -----------------------------------------------------------------
def jacobi_method():
    """todo"""

def sor_method():
    """todo"""


# x = np.zeros((20,1))

# print(is_criteria_satisfied(get_a_matrix(5), get_b_vector(5), x))
print(is_positive_definite(get_a_matrix(0.5)))