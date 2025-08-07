import numpy as np

#  Q1 (a) 
arr = np.array([1, 2, 3, 6, 4, 5])
reversed_arr = arr[::-1]
print("Q1(a) - Reversed Array:", reversed_arr)
print()

# Q1 (b) 
array1 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]])
flat1 = array1.flatten()
flat2 = array1.ravel()
print("Q1(b) - Flatten using flatten():", flat1)
print("Q1(b) - Flatten using ravel():", flat2)
print()

# Q1 (c) 
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[1, 2], [3, 4]])
comparison = np.array_equal(arr1, arr2)
print("Q1(c) - Are arrays equal?:", comparison)
print()

#  Q1 (d) 
x = np.array([1, 2, 3, 4, 5, 1, 2, 1, 1, 1])
values_x, counts_x = np.unique(x, return_counts=True)
most_freq_x = values_x[np.argmax(counts_x)]
indices_x = np.where(x == most_freq_x)
print("Q1(d) - x:")
print("Most frequent value:", most_freq_x)
print("Indices:", indices_x[0])
print()

y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3])
values_y, counts_y = np.unique(y, return_counts=True)
most_freq_y = values_y[np.argmax(counts_y)]
indices_y = np.where(y == most_freq_y)
print("Q1(d) - y:")
print("Most frequent value:", most_freq_y)
print("Indices:", indices_y[0])
print()

# Q1 (e) 
gfg = np.matrix('[4, 1, 9; 12, 3, 1; 4, 5, 6]')
total_sum = gfg.sum()
row_sum = gfg.sum(axis=1)
col_sum = gfg.sum(axis=0)
print("Q1(e):")
print("Sum of all elements:", total_sum)
print("Row-wise sum:\n", row_sum)
print("Column-wise sum:\n", col_sum)
print()

# Q1 (f) 
n_array = np.array([[55, 25, 15], [30, 44, 2], [11, 45, 77]])
diag_sum = np.trace(n_array)
eigen_values = np.linalg.eigvals(n_array)
eigen_vectors = np.linalg.eig(n_array)[1]
inverse_matrix = np.linalg.inv(n_array)
determinant = np.linalg.det(n_array)

print("Q1(f):")
print("Diagonal sum:", diag_sum)
print("Eigen values:", eigen_values)
print("Eigen vectors:\n", eigen_vectors)
print("Inverse of matrix:\n", inverse_matrix)
print("Determinant:", determinant)
print()

# Q1 (g) 
p1 = np.array([[1, 2], [2, 3]])
q1 = np.array([[4, 5], [6, 7]])
product1 = np.dot(p1, q1)
cov1 = np.cov(p1.T, q1.T)
print("Q1(g) - Case 1:")
print("Product:\n", product1)
print("Covariance:\n", cov1)
print()

p2 = np.array([[1, 2], [2, 3], [4, 5]])
q2 = np.array([[4, 5, 1], [6, 7, 2]])
product2 = np.dot(p2, q2)
cov2 = np.cov(p2.T, q2.T)
print("Q1(g) - Case 2:")
print("Product:\n", product2)
print("Covariance:\n", cov2)
print()

# Q1 (h) 
x = np.array([[2, 3, 4], [3, 2, 9]])
y = np.array([[1, 5, 0], [5, 10, 3]])

inner_product = np.inner(x, y)
print("Q1(h) - Inner Product:\n", inner_product)

outer_product = np.outer(x, y)
print("Q1(h) - Outer Product:\n", outer_product)

a = x.flatten()
b = y.flatten()
cartesian_a, cartesian_b = np.meshgrid(a, b)
cartesian_product = np.vstack([cartesian_a.ravel(), cartesian_b.ravel()]).T
print("Q1(h) - Cartesian Product:\n", cartesian_product)






#  Q2(a) 
array = np.array([[1, -2, 3], [-4, 5, -6]])

# i 
abs_array = np.abs(array)
print("Q2(a)(i) - Absolute values:\n", abs_array)

flat = array.flatten()

# ii

percentile_flat = np.percentile(flat, [25, 50, 75])

percentile_col = np.percentile(array, [25, 50, 75], axis=0)

percentile_row = np.percentile(array, [25, 50, 75], axis=1)

print("\nQ2(a)(ii) - Percentiles:")
print("Flattened (25%, 50%, 75%):", percentile_flat)
print("Column-wise percentiles:\n", percentile_col)
print("Row-wise percentiles:\n", percentile_row)

# iii
mean_flat = np.mean(flat)
median_flat = np.median(flat)
std_flat = np.std(flat)

mean_col = np.mean(array, axis=0)
median_col = np.median(array, axis=0)
std_col = np.std(array, axis=0)

mean_row = np.mean(array, axis=1)
median_row = np.median(array, axis=1)
std_row = np.std(array, axis=1)

print("\nQ2(a)(iii) - Mean, Median, Standard Deviation:")
print("Flattened - Mean:", mean_flat, "Median:", median_flat, "Std Dev:", std_flat)
print("Column-wise - Mean:", mean_col, "Median:", median_col, "Std Dev:", std_col)
print("Row-wise - Mean:", mean_row, "Median:", median_row, "Std Dev:", std_row)


#  Q2(b) 
a = np.array([-1.8, -1.6, -0.5, 0.5, 1.6, 1.8, 3.0])

floor_vals = np.floor(a)
ceil_vals = np.ceil(a)
trunc_vals = np.trunc(a)
round_vals = np.round(a)

print("\nQ2(b) - Floor, Ceil, Trunc, Round:")
print("Original:", a)
print("Floor:", floor_vals)
print("Ceil:", ceil_vals)
print("Trunc:", trunc_vals)
print("Rounded:", round_vals)






# Q3(a)
array_a = np.array([10, 52, 62, 16, 16, 54, 453])

# i
sorted_array = np.sort(array_a)
print("Q3(a)(i) - Sorted array:", sorted_array)

# ii
sorted_indices = np.argsort(array_a)
print("Q3(a)(ii) - Indices of sorted array:", sorted_indices)

# iii 
smallest_4 = np.sort(array_a)[:4]
print("Q3(a)(iii) - 4 smallest elements:", smallest_4)

# iv
largest_5 = np.sort(array_a)[-5:]
print("Q3(a)(iv) - 5 largest elements:", largest_5)

# Q3(b)
array_b = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])

# i
integer_elements = array_b[array_b == array_b.astype(int)]
print("\nQ3(b)(i) - Integer elements only:", integer_elements)

# ii
float_elements = array_b[array_b != array_b.astype(int)]
print("Q3(b)(ii) - Float elements only:", float_elements)





from PIL import Image

# Q4(a)
def img_to_array(path):
    img = Image.open(path)
    img_array = np.array(img)

    if img.mode == 'RGB':
        print("Image is in RGB format")
    elif img.mode == 'L':
        print("Image is in Grayscale format")
    else:
        print("Other format:", img.mode)

    np.savetxt("image_array.txt", img_array.reshape(-1, img_array.shape[-1]) if img_array.ndim == 3 else img_array, fmt='%d')
    print("Image array saved to image_array.txt")
    return img_array


# Q4(b) 
def load_array_from_file(file_path):
    try:
        arr = np.loadtxt(file_path, dtype=int)
        print("Loaded array shape:", arr.shape)
        return arr
    except Exception as e:
        print("Error loading file:", e)
        return None


