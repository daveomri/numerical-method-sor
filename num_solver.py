# David Omrai 23.12.2021
import numpy as np
from numpy.linalg import eig

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

def get_e_matrix():
    e_matrix = np.zeros((20,20), dtype=float)

    for i in range(0, e_matrix.shape[0]):
        e_matrix[i, i] = 1.

    return e_matrix

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

def get_l2_norm(x_vector):
    elem_sum = 0
    for i in range(0, x_vector.shape[0]):
        elem_sum += abs(x_vector[i, 0]**2)
    return elem_sum**0.5

def is_criteria_satisfied(a_matrix, b_vector, x_vector):
    return (
        get_l2_norm(
            mult_matrices(a_matrix, x_vector) - b_vector)/
        get_l2_norm(b_vector)) < 10**-6
    # return (
    #     np.linalg.norm(
    #         mult_matrices(a_matrix, x_vector) - b_vector)/
    #     np.linalg.norm(b_vector)) < 10**-6

def get_spectral_radius(matrix):
    max_eigval = 0
    for eigval in np.linalg.eigvals(matrix):
        if max_eigval < abs(eigval):
            max_eigval = eigval
    return max_eigval

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

"""
    Funkce reprezentuje jeden iteracni krok, na vstupu
    bere matici Q, ktera se lisi dle pouzite metody a
    dale provadi operace s maticemi a vektory dle vzorecku
    probiraneho na prednasce

    Postupne se provede inverze matice Q, odecet matic Q a A,
    vynasobeni tohoto rozdilu vektorem xk, prictenim vektoru b
    a nasledne vynasoveni inverze matice Q s vysledkem v zavorce

    Pouzita rovnice je v tomto tvaru
    xk+1 = Q^-1((Q - A)xk + b)
"""
def get_next_approx(q_matrix, a_matrix, x_vector, b_vector):
    return mult_matrices(inverse_matrix(q_matrix), (mult_matrices(q_matrix-a_matrix, x_vector) + b_vector))

# Iteracni metody -----------------------------------------------------------------
def jacobi_method(y, max_iter=10000):
    print("Spousteni Jacobiho metody se vstupnim parametrem {}".format(y))
    # Vytvoreni matice Q
    a_matrix = get_a_matrix(y)
    q_matrix = get_d_matrix(a_matrix)

    e_matrix = get_e_matrix()
    w_matrix = e_matrix - mult_matrices(inverse_matrix(q_matrix), a_matrix)
    wk_matrix = w_matrix
    # Vytvoreni vektoru b
    b_vector = get_b_vector(y)

    # Overeni konvergence metody
    print("Overeni konvergence metody")
    if is_diagonal_dominant(q_matrix) == True:
        print("Matice Q je radkove/sloupcove diagonalne dominantni")
        print("Jacobiho metoda bude tedy konvergovat pro libovolnou poc. aproximaci x0")
    else:
        print("Matice Q neni radkove ani sloupcove diagonalne dominantni")
        print("Jacobiho metoda muze, ale take nemusi konvergovat")

    x_cur = get_x0_vector()
    # Aproximace reseni soustavy rovnic Ax = b
    for i in range (0, max_iter):
        print("Iterace: {}".format(i))
        if (is_criteria_satisfied(a_matrix, b_vector, x_cur)):
            print("Podminka aproximace splnena v {} iterace".format(i))
            return True
        # Overeni konvergence aproximaci
        if i != 0 and get_spectral_radius(wk_matrix) > 1:
            break
        if i != 0:
            wk_matrix = mult_matrices(wk_matrix, w_matrix)

        # Vypocitej dalsi aproximaci
        x_cur = get_next_approx(q_matrix, a_matrix, x_cur, b_vector)
    
    print("Metoda diverguje, nepodarilo se uspokojit zadanou podminku aproximace")
    return False
            

def sor_method(y, max_iter=10000):
    print("Spousteni SOR metody se vstupnim parametrem {}".format(y))
    # Vytvoreni matice Q
    a_matrix = get_a_matrix(y)
    d_matrix = get_d_matrix(a_matrix)
    l_matrix = get_l_matrix(a_matrix)
    q_matrix = d_matrix + l_matrix

    e_matrix = get_e_matrix()
    w_matrix = e_matrix - mult_matrices(inverse_matrix(q_matrix), a_matrix)
    wk_matrix = w_matrix
    # Vytvoreni vektoru b
    b_vector = get_b_vector(y)

    # Overeni konvergence metody
    print("Overeni konvergence metody")
    if is_diagonal_dominant(q_matrix) == True:
        print("Matice Q je radkove/sloupcove diagonalne dominantni")
        print("SOR metoda bude tedy konvergovat pro libovolnou poc. aproximaci x0")
    else:
        print("Matice Q neni radkove ani sloupcove diagonalne dominantni")
        print("SOR metoda muze, ale take nemusi konvergovat")
    if is_positive_definite(q_matrix) == True:
        print("Matice Q je symetricka a pozitivne definitni")
        print("SOR metoda bude tedy konvergovat pro libovolnou poc. aproximaci x0")
    else:
        print("Matice Q neni symetricka a zaroven pozitivne definitni")
        print("SOR metoda muze, ale take nemusi konvergovat")

    x_cur = get_x0_vector()
    # Aproximace reseni soustavy rovnic Ax = b
    for i in range (0, max_iter):
        print("Iterace: {}".format(i))
        if (is_criteria_satisfied(a_matrix, b_vector, x_cur)):
            print("Podminka aproximace splnena v {} iterace".format(i))
            return True
        
        # Overeni konvergence aproximaci
        if i != 0 and get_spectral_radius(wk_matrix) >= 1:
            break
        if i != 0:
            wk_matrix = mult_matrices(wk_matrix, w_matrix)

        # Vypocitej dalsi aproximaci
        x_cur = get_next_approx(q_matrix, a_matrix, x_cur, b_vector)
    
    print("Metoda diverguje, nepodarilo se uspokojit zadanou podminku aproximace")
    return False
            

# Spusteni iteracnich metod nad mnozinou promennych y
# y_vals = [5, 2, 0.5]

# for y in y_vals:
#     jacobi_method(y)
#     print("---------------------------------------")
#     sor_method(y)
#     print("---------------------------------------")

jacobi_method(0.5)