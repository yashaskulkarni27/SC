
import numpy as np

# Define two fuzzy sets
A = np.array([0.2, 0.6, 0.8])
B = np.array([0.5, 0.7, 0.3])

# Fuzzy set operations
def fuzzy_union(A, B):
    return np.maximum(A, B)

def fuzzy_intersection(A, B):
    return np.minimum(A, B)

def fuzzy_complement(A):
    return 1 - A

def fuzzy_difference(A, B):
    return np.maximum(0, A - B)

def fuzzy_cartesian_product(A, B):
    return np.array([[min(a, b) for b in B] for a in A])

def fuzzy_maxmin_composition(R, S):
    return np.array([[np.max(np.minimum(R[i], S[:, j])) for j in range(S.shape[1])] for i in range(R.shape[0])])

# Compute and display fuzzy operations
union_result = fuzzy_union(A, B)
intersection_result = fuzzy_intersection(A, B)
complement_result_A = fuzzy_complement(A)
difference_result = fuzzy_difference(A, B)
cartesian_product_result = fuzzy_cartesian_product(A, B)

print("Fuzzy Union:", union_result)
print("Fuzzy Intersection:", intersection_result)
print("Fuzzy Complement of A:", complement_result_A)
print("Fuzzy Difference (A - B):", difference_result)
print("Fuzzy Cartesian Product:\n", cartesian_product_result)

# For max-min composition, we use a sample relation matrix
R = np.array([[0.2, 0.5], [0.8, 0.6]])
S = np.array([[0.7, 0.4], [0.5, 0.9]])
maxmin_composition_result = fuzzy_maxmin_composition(R, S)

print("Fuzzy Max-Min Composition:\n", maxmin_composition_result)
