# Ouroboros Library Documentation

NOTE:-AI generated documentation, may contain errors. Please refer to the source code for accurate information. And report any issues you find.

This document provides comprehensive documentation for all public APIs in the Ouroboros tensor library.

## Table of Contents

1. [Shape Class](#shape-class)
2. [Tensor Class](#tensor-class)
3. [Iterator Classes](#iterator-classes)
   - [NDRange](#ndrange)
   - [IdxIterator](#idxiterator)
   - [IdxIterator2](#idxiterator2)
4. [Bitwise Operators](#bitwise-operators)
5. [Arithmetic Operators](#arithmetic-operators)
6. [Comparison Operators](#comparison-operators)
7. [Logical Operators](#logical-operators)
8. [Utility Functions](#utility-functions)
9. [Implementation Details](#implementation-details)
   - [Shape Class Implementation](#shape-class-implementation)
   - [Iterator Classes Implementation](#iterator-classes-implementation)
   - [Logical Operators Implementation](#logical-operators-implementation-op3cpp)
10. [Comprehensive Examples](#comprehensive-examples)
   - [Shape Examples](#shape-examples)
   - [Tensor Construction and Manipulation](#tensor-construction-and-manipulation)
   - [Transform Operations](#transform-operations)
   - [Reduce Operations](#reduce-operations)
   - [Accumulate Operations](#accumulate-operations)
   - [Outer Product](#outer-product)
   - [Broadcasting](#broadcasting)
   - [Concatenation](#concatenation)
   - [Flip and Transpose](#flip-and-transpose)
   - [Boolean Tensor Operations](#boolean-tensor-operations)
   - [Complete Workflow Example](#complete-workflow-example)
11. [Performance Notes](#performance-notes)
12. [Memory Management](#memory-management)
13. [Best Practices and Common Patterns](#best-practices-and-common-patterns)
   - [Choosing Between Compile-Time and Runtime Function Parameters](#choosing-between-compile-time-and-runtime-function-parameters)
   - [Efficient Iteration Patterns](#efficient-iteration-patterns)
   - [Memory Management Best Practices](#memory-management-best-practices)
   - [Avoiding Common Pitfalls](#avoiding-common-pitfalls)
   - [Optimization Tips](#optimization-tips)
   - [Integration with Standard Library](#integration-with-standard-library)
14. [Notes](#notes)

---

## Shape Class

The `Shape` class represents the dimensions of a tensor. It stores the shape information, strides, and handles indexing calculations.

### Constructors

```cpp
Shape(std::size_t dim, std::size_t val);
```
Creates a shape with `dim` dimensions, each having size `val`.

**Example:**
```cpp
Shape s(3, 5);  // Creates shape [5, 5, 5]
```

**NumPy equivalent:** `np.ones(3) * 5` (for shape specification)

---

```cpp
Shape(std::size_t dim, std::size_t* shape);
```
Creates a shape from an array. The array is copied.

**Parameters:**
- `dim`: Number of dimensions
- `shape`: Pointer to array of dimension sizes

**Example:**
```cpp
std::size_t dims[] = {2, 3, 4};
Shape s(3, dims);  // Creates shape [2, 3, 4]
```

---

```cpp
Shape(std::initializer_list<std::size_t> shape);
```
Creates a shape from an initializer list.

**Example:**
```cpp
Shape s({2, 3, 4});  // Creates shape [2, 3, 4]
```

**NumPy equivalent:** Passing shape tuple like `(2, 3, 4)` to array creation functions

---

```cpp
Shape(std::vector<std::size_t> shape);
```
Creates a shape from a vector.

**Example:**
```cpp
std::vector<std::size_t> dims = {2, 3, 4};
Shape s(dims);  // Creates shape [2, 3, 4]
```

---

```cpp
Shape(const Shape& shape);
```
Copy constructor.

---

```cpp
Shape(Shape&& shape);
```
Move constructor.

### Operators

```cpp
void operator=(const Shape& shape);
void operator=(Shape&& shape);
void operator=(std::initializer_list<std::size_t> shape);
```
Assignment operators for copying or moving shape data.

---

```cpp
const std::size_t operator[](std::size_t index) const;
```
Accesses the size of dimension at `index`.

**Example:**
```cpp
Shape s({2, 3, 4});
std::cout << s[1];  // Outputs: 3
```

**NumPy equivalent:** `arr.shape[1]`

---

```cpp
bool operator==(const Shape& shape) const;
bool operator!=(const Shape& shape) const;
```
Compares two shapes for equality.

### Methods

```cpp
void set(std::size_t index, std::size_t val);
```
Sets the size of dimension at `index` to `val`. Automatically recomputes strides and count.

**Example:**
```cpp
Shape s({2, 3, 4});
s.set(1, 5);  // Shape becomes [2, 5, 4]
```

---

```cpp
const std::size_t get_stride(std::size_t index) const;
```
Returns the stride for dimension at `index`. Strides determine memory layout.

**NumPy equivalent:** `arr.strides[index]` (but in element count, not bytes)

---

```cpp
std::size_t offset(const std::vector<std::size_t>& indices) const;
std::size_t offset(const std::size_t* start) const;
```
Computes the linear offset in memory for given multi-dimensional indices.

**Example:**
```cpp
Shape s({2, 3, 4});
std::vector<std::size_t> idx = {1, 2, 3};
std::size_t off = s.offset(idx);  // Computes linear offset for position [1,2,3]
```

---

```cpp
const std::size_t* begin() const;
const std::size_t* end() const;
```
Returns pointers for iterating over shape dimensions.

---

```cpp
std::size_t count() const;
```
Returns the total number of elements (product of all dimensions).

**Example:**
```cpp
Shape s({2, 3, 4});
std::cout << s.count();  // Outputs: 24
```

**NumPy equivalent:** `arr.size`

---

```cpp
std::size_t dim() const;
```
Returns the number of dimensions.

**Example:**
```cpp
Shape s({2, 3, 4});
std::cout << s.dim();  // Outputs: 3
```

**NumPy equivalent:** `arr.ndim`

---

```cpp
std::vector<std::size_t> to_vector() const;
```
Converts the shape to a vector.

---

```cpp
std::ostream& operator<<(std::ostream& os, const Shape& shape);
```
Stream output operator for printing shapes.

**Example:**
```cpp
Shape s({2, 3, 4});
std::cout << s;  // Outputs shape information
```

---

## Tensor Class

The `Tensor<T>` template class represents an N-dimensional array. It's the core data structure of the library.

### Constructors

```cpp
Tensor(const Shape& shape);
```
Creates a tensor with given shape. Elements are uninitialized.

**Example:**
```cpp
Tensor<double> t(Shape({2, 3}));  // 2x3 tensor of doubles
```

**NumPy equivalent:** `np.empty((2, 3), dtype=float)`

---

```cpp
Tensor(const Shape& shape, T val);
```
Creates a tensor with given shape, filled with value `val`.

**Example:**
```cpp
Tensor<double> t(Shape({2, 3}), 1.0);  // 2x3 tensor filled with 1.0
```

**NumPy equivalent:** `np.full((2, 3), 1.0)` or `np.ones((2, 3))`

---

```cpp
Tensor(const Shape& shape, T* data);
```
Creates a tensor using existing data pointer. The tensor takes ownership of the data.

**Warning:** The data pointer should be allocated with `new[]` as it will be deleted with `delete[]`.

---

```cpp
Tensor(const Tensor<T>& tensor);
```
Copy constructor. Creates a deep copy of the tensor.

**Example:**
```cpp
Tensor<double> t1({2, 3}, 1.0);
Tensor<double> t2(t1);  // Deep copy
```

**NumPy equivalent:** `arr2 = arr1.copy()`

---

```cpp
Tensor(Tensor<T>&& tensor);
```
Move constructor. Takes ownership of tensor data without copying.

### Operators

```cpp
void operator=(const Tensor<T>& tensor);
```
Copy assignment operator.

---

```cpp
void operator=(Tensor<T>&& tensor);
```
Move assignment operator.

---

```cpp
T& operator[](std::size_t index);
const T& operator[](std::size_t index) const;
```
Accesses element at linear index.

**Example:**
```cpp
Tensor<double> t({2, 3}, 1.0);
t[0] = 5.0;  // Sets first element to 5.0
```

**NumPy equivalent:** `arr.flat[index]` or `arr.ravel()[index]`

---

```cpp
T& operator[](const std::vector<std::size_t>& indices);
const T& operator[](const std::vector<std::size_t>& indices) const;
T& operator[](const std::size_t* indices);
const T& operator[](const std::size_t* indices) const;
```
Accesses element at multi-dimensional indices.

**Example:**
```cpp
Tensor<double> t({2, 3}, 1.0);
t[{1, 2}] = 5.0;  // Sets element at position [1,2] to 5.0
```

**NumPy equivalent:** `arr[1, 2] = 5.0`

### Methods

```cpp
void reshape(const Shape& shape);
```
Changes the shape of the tensor without modifying data. Total element count must remain the same.

**Example:**
```cpp
Tensor<double> t({2, 3}, 1.0);
t.reshape(Shape({3, 2}));  // Reshape to 3x2
```

**NumPy equivalent:** `arr.reshape((3, 2))`

---

```cpp
const Shape& shape() const;
```
Returns the shape of the tensor.

**Example:**
```cpp
Tensor<double> t({2, 3}, 1.0);
const Shape& s = t.shape();
```

**NumPy equivalent:** `arr.shape`

---

```cpp
const T* data() const;
T* data();
```
Returns pointer to the underlying data array.

**Example:**
```cpp
Tensor<double> t({2, 3}, 1.0);
double* ptr = t.data();
```

**NumPy equivalent:** `arr.data` or `arr.ctypes.data`

---

```cpp
const std::size_t size() const;
```
Returns the total number of elements in the tensor.

**Example:**
```cpp
Tensor<double> t({2, 3}, 1.0);
std::cout << t.size();  // Outputs: 6
```

**NumPy equivalent:** `arr.size`

---

```cpp
const std::size_t dim() const;
```
Returns the number of dimensions.

**Example:**
```cpp
Tensor<double> t({2, 3}, 1.0);
std::cout << t.dim();  // Outputs: 2
```

**NumPy equivalent:** `arr.ndim`

---

```cpp
std::size_t offset(const std::vector<std::size_t>& indices) const;
```
Computes linear offset for given multi-dimensional indices.

---

```cpp
template<T(*func)(T*, T*)>
T reduce() const;
```
Applies a reduction function over all elements.

**Example:**
```cpp
Tensor<double> t({2, 3}, 2.0);
// Define reduction function
double sum_func(double* begin, double* end) {
    return std::accumulate(begin, end, 0.0);
}
double total = t.reduce<sum_func>();
```

**NumPy equivalent:** Reduction operations like `arr.sum()`, `arr.max()`, etc.

---

```cpp
std::ostream& operator<<(std::ostream& os, const Tensor<U>& tensor);
```
Stream output operator for printing tensors.

**Example:**
```cpp
Tensor<double> t({2, 3}, 1.0);
std::cout << t;  // Prints tensor in nested array format
```

**NumPy equivalent:** `print(arr)`

---

## Iterator Classes

### NDRange

`NDRange` provides iteration over a tensor with certain axes fixed and others varying. Useful for operations along specific dimensions.

```cpp
NDRange(Shape s, std::unordered_set<size_t> axis);
```
Creates an NDRange iterator.

**Parameters:**
- `s`: Shape of the tensor
- `axis`: Set of axes to fix first during iteration

**Example:**
```cpp
Shape s({3, 4, 5});
std::unordered_set<size_t> axes = {1};
NDRange ndr(s, axes);
// Iterates fixing axis 1, varying others
```

---

### IdxIterator

`IdxIterator` provides iteration over tensor indices with optional fixed dimensions.

```cpp
IdxIterator(Shape s, std::unordered_map<std::size_t, std::size_t> fixed_indices = {});
```
Creates an index iterator.

**Parameters:**
- `s`: Shape of the tensor
- `fixed_indices`: Map of axis -> fixed value for that axis

**Example:**
```cpp
Shape s({3, 4, 5});
std::unordered_map<std::size_t, std::size_t> fixed = {{1, 2}};
IdxIterator itr(s, fixed);
// Iterates over indices with axis 1 fixed at 2
for(auto it = itr.begin(); it != itr.end(); ++it) {
    std::size_t offset = *it;
    auto indices = it.get_index();
    // indices[1] will always be 2
}
```

**NumPy equivalent:** Similar to `np.ndindex()` with slicing

---

### IdxIterator2

`IdxIterator2` provides iteration over a range with start, end, and step values per dimension.

```cpp
IdxIterator2(const Shape& shape_, const std::vector<size_t>& start_,
             const std::vector<size_t>& end_, const std::vector<size_t>& step_);
```
Creates a range-based index iterator.

**Parameters:**
- `shape_`: Shape of the tensor
- `start_`: Starting index for each dimension
- `end_`: Ending index for each dimension (exclusive)
- `step_`: Step size for each dimension

**Example:**
```cpp
Shape s({10, 10});
std::vector<size_t> start = {0, 0};
std::vector<size_t> end = {5, 5};
std::vector<size_t> step = {2, 1};
IdxIterator2 itr(s, start, end, step);
// Iterates: [0,0], [0,1], ..., [0,4], [2,0], [2,1], ..., [4,4]
```

**NumPy equivalent:** Similar to advanced indexing with ranges

---

## Bitwise Operators

These operators perform element-wise bitwise operations on tensors. (Defined in `op0.hpp`)

### Tensor-Tensor Operations

```cpp
Tensor<T> operator~(const Tensor<T>& a);
```
Bitwise NOT operation.

**Example:**
```cpp
Tensor<int> t({2, 2}, 5);
Tensor<int> result = ~t;
```

**NumPy equivalent:** `np.bitwise_not(arr)` or `~arr`

---

```cpp
Tensor<T> operator|(const Tensor<T>& a, const Tensor<T>& b);
```
Element-wise bitwise OR.

**NumPy equivalent:** `np.bitwise_or(a, b)` or `a | b`

---

```cpp
Tensor<T> operator&(const Tensor<T>& a, const Tensor<T>& b);
```
Element-wise bitwise AND.

**NumPy equivalent:** `np.bitwise_and(a, b)` or `a & b`

---

```cpp
Tensor<T> operator^(const Tensor<T>& a, const Tensor<T>& b);
```
Element-wise bitwise XOR.

**NumPy equivalent:** `np.bitwise_xor(a, b)` or `a ^ b`

---

```cpp
Tensor<T> operator<<(const Tensor<T>& a, const Tensor<T>& b);
```
Element-wise left shift.

**NumPy equivalent:** `np.left_shift(a, b)` or `a << b`

---

```cpp
Tensor<T> operator>>(const Tensor<T>& a, const Tensor<T>& b);
```
Element-wise right shift.

**NumPy equivalent:** `np.right_shift(a, b)` or `a >> b`

### Tensor-Scalar Operations

```cpp
Tensor<T> operator|(const Tensor<T>& a, T b);
Tensor<T> operator|(T a, const Tensor<T>& b);
```
Bitwise OR with scalar.

**NumPy equivalent:** `arr | scalar`

---

```cpp
Tensor<T> operator&(const Tensor<T>& a, T b);
Tensor<T> operator&(T a, const Tensor<T>& b);
```
Bitwise AND with scalar.

**NumPy equivalent:** `arr & scalar`

---

```cpp
Tensor<T> operator^(const Tensor<T>& a, T b);
Tensor<T> operator^(T a, const Tensor<T>& b);
```
Bitwise XOR with scalar.

**NumPy equivalent:** `arr ^ scalar`

---

```cpp
Tensor<T> operator<<(const Tensor<T>& a, T b);
Tensor<T> operator<<(T a, const Tensor<T>& b);
```
Left shift with scalar.

**NumPy equivalent:** `arr << scalar`

---

```cpp
Tensor<T> operator>>(const Tensor<T>& a, T b);
Tensor<T> operator>>(T a, const Tensor<T>& b);
```
Right shift with scalar.

**NumPy equivalent:** `arr >> scalar`

---

## Arithmetic Operators

These operators perform element-wise arithmetic operations. (Defined in `op1.hpp`)

### Unary Operations

```cpp
Tensor<T> operator-(const Tensor<T>& a);
```
Negation operator.

**Example:**
```cpp
Tensor<double> t({2, 2}, 5.0);
Tensor<double> result = -t;  // All elements become -5.0
```

**NumPy equivalent:** `-arr` or `np.negative(arr)`

### Tensor-Tensor Operations

```cpp
Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b);
```
Element-wise addition.

**Example:**
```cpp
Tensor<double> t1({2, 2}, 1.0);
Tensor<double> t2({2, 2}, 2.0);
Tensor<double> result = t1 + t2;  // Elements: 3.0
```

**NumPy equivalent:** `a + b` or `np.add(a, b)`

---

```cpp
Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b);
```
Element-wise subtraction.

**NumPy equivalent:** `a - b` or `np.subtract(a, b)`

---

```cpp
Tensor<T> operator*(const Tensor<T>& a, const Tensor<T>& b);
```
Element-wise multiplication (Hadamard product).

**NumPy equivalent:** `a * b` or `np.multiply(a, b)`

---

```cpp
Tensor<T> operator/(const Tensor<T>& a, const Tensor<T>& b);
```
Element-wise division.

**NumPy equivalent:** `a / b` or `np.divide(a, b)`

### Tensor-Scalar Operations

```cpp
Tensor<T> operator+(const Tensor<T>& a, T b);
Tensor<T> operator+(T a, const Tensor<T>& b);
```
Addition with scalar.

**NumPy equivalent:** `arr + scalar`

---

```cpp
Tensor<T> operator-(const Tensor<T>& a, T b);
Tensor<T> operator-(T a, const Tensor<T>& b);
```
Subtraction with scalar.

**NumPy equivalent:** `arr - scalar` or `scalar - arr`

---

```cpp
Tensor<T> operator*(const Tensor<T>& a, T b);
Tensor<T> operator*(T a, const Tensor<T>& b);
```
Multiplication with scalar.

**NumPy equivalent:** `arr * scalar`

---

```cpp
Tensor<T> operator/(const Tensor<T>& a, T b);
Tensor<T> operator/(T a, const Tensor<T>& b);
```
Division with scalar.

**NumPy equivalent:** `arr / scalar` or `scalar / arr`

### In-place Operations

```cpp
void operator+=(Tensor<T>& a, const Tensor<T>& b);
```
In-place addition.

**NumPy equivalent:** `a += b`

---

```cpp
void operator-=(Tensor<T>& a, const Tensor<T>& b);
```
In-place subtraction.

**NumPy equivalent:** `a -= b`

---

```cpp
void operator*=(Tensor<T>& a, const Tensor<T>& b);
```
In-place multiplication.

**NumPy equivalent:** `a *= b`

---

```cpp
void operator/=(Tensor<T>& a, const Tensor<T>& b);
```
In-place division.

**NumPy equivalent:** `a /= b`

---

```cpp
void operator+=(Tensor<T>& a, T b);
void operator-=(Tensor<T>& a, T b);
void operator*=(Tensor<T>& a, T b);
void operator/=(Tensor<T>& a, T b);
```
In-place operations with scalars.

**NumPy equivalent:** `a += scalar`, etc.

### Matrix Operations

```cpp
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b);
```
Matrix multiplication using BLAS. Only for 2D tensors. Supports `float` and `double` types.

**Example:**
```cpp
Tensor<double> a({2, 3}, 1.0);
Tensor<double> b({3, 4}, 2.0);
Tensor<double> result = matmul(a, b);  // Shape: [2, 4]
```

**NumPy equivalent:** `np.matmul(a, b)` or `a @ b`

---

```cpp
Tensor<T> matvecmul(const Tensor<T>& a, const Tensor<T>& b);
```
Matrix-vector multiplication using BLAS. `a` is 2D, `b` is 1D. Supports `float` and `double` types.

**Example:**
```cpp
Tensor<double> a({2, 3}, 1.0);
Tensor<double> b({3}, 2.0);
Tensor<double> result = matvecmul(a, b);  // Shape: [2]
```

**NumPy equivalent:** `np.matmul(a, b)` where `b` is 1D

---

## Comparison Operators

These operators perform element-wise comparisons, returning boolean tensors. (Defined in `op2.hpp`)

### Tensor-Tensor Comparisons

```cpp
Tensor<bool> operator==(const Tensor<T>& a, const Tensor<T>& b);
```
Element-wise equality comparison.

**Example:**
```cpp
Tensor<double> t1({2, 2}, 1.0);
Tensor<double> t2({2, 2}, 1.0);
Tensor<bool> result = (t1 == t2);  // All elements true
```

**NumPy equivalent:** `a == b` or `np.equal(a, b)`

---

```cpp
Tensor<bool> operator!=(const Tensor<T>& a, const Tensor<T>& b);
```
Element-wise inequality comparison.

**NumPy equivalent:** `a != b` or `np.not_equal(a, b)`

---

```cpp
Tensor<bool> operator<(const Tensor<T>& a, const Tensor<T>& b);
```
Element-wise less-than comparison.

**NumPy equivalent:** `a < b` or `np.less(a, b)`

---

```cpp
Tensor<bool> operator>(const Tensor<T>& a, const Tensor<T>& b);
```
Element-wise greater-than comparison.

**NumPy equivalent:** `a > b` or `np.greater(a, b)`

---

```cpp
Tensor<bool> operator<=(const Tensor<T>& a, const Tensor<T>& b);
```
Element-wise less-than-or-equal comparison.

**NumPy equivalent:** `a <= b` or `np.less_equal(a, b)`

---

```cpp
Tensor<bool> operator>=(const Tensor<T>& a, const Tensor<T>& b);
```
Element-wise greater-than-or-equal comparison.

**NumPy equivalent:** `a >= b` or `np.greater_equal(a, b)`

### Tensor-Scalar Comparisons

```cpp
Tensor<bool> operator==(const Tensor<T>& a, double b);
Tensor<bool> operator==(double a, const Tensor<T>& b);
```
Equality comparison with scalar.

**NumPy equivalent:** `arr == scalar`

---

```cpp
Tensor<bool> operator!=(const Tensor<T>& a, double b);
Tensor<bool> operator!=(double a, const Tensor<T>& b);
```
Inequality comparison with scalar.

**NumPy equivalent:** `arr != scalar`

---

```cpp
Tensor<bool> operator<(const Tensor<T>& a, double b);
Tensor<bool> operator<(double a, const Tensor<T>& b);
```
Less-than comparison with scalar.

**NumPy equivalent:** `arr < scalar` or `scalar < arr`

---

```cpp
Tensor<bool> operator>(const Tensor<T>& a, double b);
Tensor<bool> operator>(double a, const Tensor<T>& b);
```
Greater-than comparison with scalar.

**NumPy equivalent:** `arr > scalar` or `scalar > arr`

---

```cpp
Tensor<bool> operator<=(const Tensor<T>& a, double b);
Tensor<bool> operator<=(double a, const Tensor<T>& b);
```
Less-than-or-equal comparison with scalar.

**NumPy equivalent:** `arr <= scalar` or `scalar <= arr`

---

```cpp
Tensor<bool> operator>=(const Tensor<T>& a, double b);
Tensor<bool> operator>=(double a, const Tensor<T>& b);
```
Greater-than-or-equal comparison with scalar.

**NumPy equivalent:** `arr >= scalar` or `scalar >= arr`

---

## Logical Operators

These operators perform element-wise logical operations on boolean tensors. (Defined in `op3.hpp`)

```cpp
Tensor<bool> operator!(const Tensor<bool>& a);
```
Logical NOT operation.

**NumPy equivalent:** `~arr` or `np.logical_not(arr)` for boolean arrays

---

```cpp
Tensor<bool> operator==(const Tensor<bool>& a, const Tensor<bool>& b);
Tensor<bool> operator!=(const Tensor<bool>& a, const Tensor<bool>& b);
```
Element-wise equality/inequality for boolean tensors.

**NumPy equivalent:** `a == b` or `a != b`

---

```cpp
Tensor<bool> operator&&(const Tensor<bool>& a, const Tensor<bool>& b);
```
Element-wise logical AND.

**NumPy equivalent:** `np.logical_and(a, b)`

---

```cpp
Tensor<bool> operator||(const Tensor<bool>& a, const Tensor<bool>& b);
```
Element-wise logical OR.

**NumPy equivalent:** `np.logical_or(a, b)`

### Boolean Tensor-Scalar Operations

```cpp
Tensor<bool> operator==(const Tensor<bool>& a, bool b);
Tensor<bool> operator==(bool a, const Tensor<bool>& b);
Tensor<bool> operator!=(const Tensor<bool>& a, bool b);
Tensor<bool> operator!=(bool a, const Tensor<bool>& b);
```
Equality/inequality comparison with boolean scalar.

---

```cpp
Tensor<bool> operator&&(const Tensor<bool>& a, bool b);
Tensor<bool> operator&&(bool a, const Tensor<bool>& b);
```
Logical AND with boolean scalar.

**NumPy equivalent:** `np.logical_and(arr, scalar)`

---

```cpp
Tensor<bool> operator||(const Tensor<bool>& a, bool b);
Tensor<bool> operator||(bool a, const Tensor<bool>& b);
```
Logical OR with boolean scalar.

**NumPy equivalent:** `np.logical_or(arr, scalar)`

---

## Utility Functions

These are higher-level functions for tensor manipulation. (Defined in `func0.hpp`)

### transform

```cpp
template<const auto func, typename T, typename... Ts>
auto transform(const T& t, const Ts&... args);
```
Applies a function element-wise to one or more tensors.

**Parameters:**
- `func`: Function to apply (must be a compile-time constant)
- `t`: Primary tensor
- `args`: Additional tensors (optional)

**Example:**
```cpp
auto square = [](double x) { return x * x; };
Tensor<double> t({2, 2}, 3.0);
auto result = transform<square>(t);  // All elements become 9.0
```

**NumPy equivalent:** `np.vectorize(func)(arr)` or using ufuncs

---

### reduce

```cpp
template<const auto func, typename T, typename... Ts>
auto reduce(size_t axis, const T& t, const Ts&... args);
```
Reduces a tensor along a specified axis.

**Parameters:**
- `func`: Reduction function taking `(IdxIterator, T* data, ...)`
- `axis`: Axis along which to reduce
- `t`: Tensor to reduce
- `args`: Additional tensors (optional)

**Example:**
```cpp
auto sum_func = [](IdxIterator itr, const double* data) {
    double sum = 0.0;
    for(auto it = itr.begin(); it != itr.end(); ++it) {
        sum += data[*it];
    }
    return sum;
};
Tensor<double> t({3, 4}, 1.0);
auto result = reduce<sum_func>(1, t);  // Sum along axis 1, shape becomes [3, 1]
```

**NumPy equivalent:** `np.sum(arr, axis=1, keepdims=True)` or other reduction functions

---

### accumulate

```cpp
template<const auto func, typename T>
auto accumulate(const T& t, std::size_t axis = 0, double initial = 0);
```
Computes cumulative operation along an axis.

**Parameters:**
- `func`: Accumulation function
- `t`: Input tensor
- `axis`: Axis along which to accumulate (default: 0)
- `initial`: Initial value for accumulation (default: 0)

**Example:**
```cpp
auto add = [](double a, double b) { return a + b; };
Tensor<double> t({3, 3}, 1.0);
auto result = accumulate<add>(t, 0, 0.0);  // Cumulative sum along axis 0
```

**NumPy equivalent:** `np.cumsum(arr, axis=0)`

---

### outer

```cpp
template<const auto func, typename T0, typename T1>
auto outer(const T0& a, const T1& b);
```
Computes outer product of two tensors using a custom function.

**Parameters:**
- `func`: Function to apply between elements
- `a`: First tensor
- `b`: Second tensor

**Example:**
```cpp
auto multiply = [](double x, double y) { return x * y; };
Tensor<double> a({2, 3}, 2.0);
Tensor<double> b({3, 2}, 3.0);
auto result = outer<multiply>(a, b);  // Shape: [2, 3, 3, 2]
```

**NumPy equivalent:** `np.outer(a.ravel(), b.ravel()).reshape(...)` or broadcasting operations

---

### concat

```cpp
template<typename T0, typename... Ts>
T0 concat(std::size_t axis, const T0& t0, const T0& t1, const Ts&... ts);
```
Concatenates tensors along a specified axis.

**Parameters:**
- `axis`: Axis along which to concatenate
- `t0, t1, ts...`: Tensors to concatenate

**Example:**
```cpp
Tensor<double> t1({2, 3}, 1.0);
Tensor<double> t2({2, 3}, 2.0);
auto result = concat(0, t1, t2);  // Shape: [4, 3]
```

**NumPy equivalent:** `np.concatenate([t1, t2], axis=0)`

---

### transpose

```cpp
template<typename T>
T transpose(const T& t, std::size_t ax1 = 0, std::size_t ax2 = 1);
```
Transposes two axes of a tensor.

**Parameters:**
- `t`: Input tensor
- `ax1`: First axis to swap (default: 0)
- `ax2`: Second axis to swap (default: 1)

**Example:**
```cpp
Tensor<double> t({2, 3, 4}, 1.0);
auto result = transpose(t, 0, 2);  // Shape becomes [4, 3, 2]
```

**NumPy equivalent:** `np.transpose(arr, (2, 1, 0))` or `np.swapaxes(arr, 0, 2)`

---

### flip

```cpp
template<typename T>
T flip(const T& t, std::size_t axis);
```
Reverses the order of elements along an axis.

**Parameters:**
- `t`: Input tensor
- `axis`: Axis along which to flip

**Example:**
```cpp
Tensor<double> t({3, 3}, 1.0);
// Assuming elements differ: [[1,2,3], [4,5,6], [7,8,9]]
auto result = flip(t, 1);  // Flips columns: [[3,2,1], [6,5,4], [9,8,7]]
```

**NumPy equivalent:** `np.flip(arr, axis=1)`

---

### broadcast

```cpp
template<const auto func, typename T0, typename T1, typename... Ts>
auto broadcast(const T0& t0, const T1& t1, const Ts&... args);
```
Applies a function with broadcasting rules (similar to NumPy).

**Parameters:**
- `func`: Function to apply
- `t0, t1, args...`: Tensors to broadcast and combine

**Example:**
```cpp
auto add = [](double x, double y) { return x + y; };
Tensor<double> t1({3, 1}, 1.0);
Tensor<double> t2({1, 4}, 2.0);
auto result = broadcast<add>(t1, t2);  // Shape: [3, 4], broadcasting applied
```

**NumPy equivalent:** Automatic broadcasting in `a + b` where shapes are compatible

**Broadcasting Rules:**
- Dimensions are aligned from right to left
- Size 1 dimensions are stretched to match larger dimensions
- Other dimensions must match exactly

---

## Implementation Details

### Shape Class Implementation

The Shape class uses dynamic memory allocation to store dimension sizes and computed strides:

**Memory Layout:**
```cpp
// Internal representation
std::size_t m_dim;      // Number of dimensions
std::size_t m_count;    // Total element count (product of all dimensions)
std::size_t* m_shape;   // Array of dimension sizes
std::size_t* m_strides; // Array of strides for each dimension
```

**Stride Calculation:**
Strides are computed automatically in row-major (C-style) order:
- The last dimension has stride 1
- Each previous dimension's stride = next dimension's stride × next dimension's size
- Example: Shape `{2, 3, 4}` produces strides `{12, 4, 1}`

This allows efficient conversion from multi-dimensional indices to linear memory offsets:
```cpp
offset = index[0] * stride[0] + index[1] * stride[1] + ... + index[n-1] * stride[n-1]
```

**Memory Management:**
- Copy constructor performs deep copy of shape and stride arrays
- Move constructor transfers ownership without copying
- Assignment operators handle memory reallocation when dimensions change
- Destructor ensures proper cleanup of dynamic arrays

### Iterator Classes Implementation

#### NDRange Implementation

`NDRange` provides nested iteration over tensor dimensions with fixed and varying axes:

**Internal Structure:**
- Splits axes into two groups: fixed (outer) and varying (inner)
- Outer iterator (`Iterator0`) iterates over fixed axes
- Inner range (`Range`) iterates over varying axes at each outer position
- Uses two-level iteration for efficient memory access patterns

**Algorithm:**
1. Dimensions are partitioned based on the `axis` parameter
2. Outer loop increments through fixed axes
3. For each outer position, inner range provides iteration over remaining axes
4. Offset calculations use pre-computed strides for performance

#### IdxIterator Implementation

`IdxIterator` enables iteration with specific dimensions locked to fixed values:

**Fixed Index Handling:**
- Uses `vector<int64_t> is_fixed` where `-1` means free, non-negative means fixed
- Fixed indices contribute to the initial offset
- Only non-fixed dimensions increment during iteration

**Iteration Pattern:**
- Similar to odometer: rightmost (last) dimension increments fastest
- When a dimension reaches its limit, it resets to 0 and carries to the next
- Fixed dimensions are skipped during increment
- End state is marked with `end_flag` for efficient comparison

#### IdxIterator2 Implementation

`IdxIterator2` provides strided iteration with start, end, and step per dimension:

**Range-Based Iteration:**
- Each dimension has independent start, end (exclusive), and step values
- Iteration follows: `index[i] = start[i], start[i] + step[i], ..., < end[i]`
- Useful for implementing slicing and strided access patterns

**Example:**
```cpp
// Iterate over even indices in first dimension, all indices in second
Shape s({10, 5});
IdxIterator2 iter(s, {0, 0}, {10, 5}, {2, 1});
// Visits: [0,0], [0,1], ..., [0,4], [2,0], ..., [8,4]
```

### Logical Operators Implementation (op3.cpp)

Boolean tensor operations are implemented with simple element-wise loops:

**Design Principles:**
- All operations create new result tensors (no in-place modification)
- Element-wise operations use linear indexing for cache efficiency
- Tensor-scalar operations are commutative where mathematically appropriate

**Implementation Pattern:**
```cpp
// Example: operator&&
Tensor<bool> operator&&(const Tensor<bool>& a, const Tensor<bool>& b) {
    Tensor<bool> result(a.shape());
    for(std::size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] && b[i];
    }
    return result;
}
```

**Scalar Operations:**
- `operator==(Tensor, scalar)` and `operator==(scalar, Tensor)` both call the same implementation
- Similarly for `!=`, `&&`, and `||`
- This ensures symmetric behavior for commutative operations

## Comprehensive Examples

### Shape Examples

```cpp
#include <ouroboros/tensor.hpp>

// Initialize shapes in multiple ways
Ouroboros::Shape shape1(3, 2);           // [2, 2, 2]
Ouroboros::Shape shape2 = {1, 2, 3};     // [1, 2, 3]
Ouroboros::Shape shape3(shape2);         // Copy constructor

std::size_t array[] = {1, 2, 3};
Ouroboros::Shape shape4(3, array);       // From array

// Modify shapes
shape1.set(0, 5);                        // Changes to [5, 2, 2]

// Iterate over shape dimensions
for(auto dim : shape1) {
    std::cout << dim << " ";
}

// Get total elements and dimensions
std::cout << shape1.count();             // 20 (5*2*2)
std::cout << shape1.dim();               // 3
```

### Tensor Construction and Manipulation

```cpp
#include <ouroboros/tensor.hpp>

// Various tensor constructors
Ouroboros::Tensor<double> t1({2, 3});           // Uninitialized
Ouroboros::Tensor<double> t2({2, 3}, 1.0);      // Filled with 1.0

// Using pre-allocated data (transfers ownership)
double* data = new double[6];
for(int i = 0; i < 6; i++) data[i] = i;
Ouroboros::Tensor<double> t3({2, 3}, data);

// Copy and move
Ouroboros::Tensor<double> t4(t2);               // Deep copy
Ouroboros::Tensor<double> t5(std::move(t4));    // Move (t4 is now empty)

// Reshape (must preserve element count)
t2.reshape({3, 2});                             // 2x3 -> 3x2

// Indexing
t2[0] = 5.0;                                    // Linear indexing
t2[{1, 1}] = 3.0;                               // Multi-dimensional indexing
std::vector<size_t> idx = {0, 1};
t2[idx] = 7.0;                                  // Using vector

// Get offset for multi-dimensional index
size_t offset = t2.shape().offset({1, 1});
```

### Transform Operations

```cpp
#include <ouroboros/tensor.hpp>
#include <cmath>

// Single-argument transform
Ouroboros::Tensor<double> t({2, 2}, 2.0);
auto square = [](double x) { return x * x; };
auto result = Ouroboros::transform<square>(t);  // All elements become 4.0

// Multi-argument transform
auto power = [](double x, double y) { return std::pow(x, y); };
Ouroboros::Tensor<double> base({2, 2}, 2.0);
Ouroboros::Tensor<double> exp({2, 2}, 3.0);
auto powered = Ouroboros::transform<power>(base, exp);  // 2^3 = 8.0

// Transform with non-const function (use runtime version)
double factor = 10.0;
auto multiply = [factor](double x) { return x * factor; };
auto scaled = Ouroboros::transform(multiply, t);  // Note: no template parameter
```

### Reduce Operations

```cpp
#include <ouroboros/tensor.hpp>

Ouroboros::Tensor<double> t({2, 3});
for(size_t i = 0; i < t.size(); i++) {
    t[i] = i + 1;  // [[1, 2, 3], [4, 5, 6]]
}

// Sum reduction along axis 0
auto sum_func = [](Ouroboros::IdxIterator itr, const double* data) {
    double sum = 0.0;
    for(auto it = itr.begin(); it != itr.end(); ++it) {
        sum += data[*it];
    }
    return sum;
};
auto result = Ouroboros::reduce<sum_func>(0, t);  // Shape: [1, 3], values: [5, 7, 9]

// Max reduction along axis 1
auto max_func = [](Ouroboros::IdxIterator itr, const double* data) {
    double max_val = -std::numeric_limits<double>::infinity();
    for(auto it = itr.begin(); it != itr.end(); ++it) {
        max_val = std::max(max_val, data[*it]);
    }
    return max_val;
};
auto max_result = Ouroboros::reduce<max_func>(1, t);  // Shape: [2, 1], values: [3, 6]
```

### Accumulate Operations

```cpp
#include <ouroboros/tensor.hpp>

Ouroboros::Tensor<double> t({2, 2});
for(size_t i = 0; i < t.size(); i++) {
    t[i] = i + 1;  // [[1, 2], [3, 4]]
}

// Cumulative sum along axis 0
auto add = [](double acc, double x) { return acc + x; };
auto cumsum = Ouroboros::accumulate<add>(t, 0, 0.0);
// Result: [[1, 2], [4, 6]]

// Cumulative product along axis 1 with initial value 1
auto multiply = [](double acc, double x) { return acc * x; };
auto cumprod = Ouroboros::accumulate<multiply>(t, 1, 1.0);
// Result: [[1, 2], [3, 12]]
```

### Outer Product

```cpp
#include <ouroboros/tensor.hpp>

// Outer product combines every element from two tensors
Ouroboros::Tensor<double> t1({2, 2});
Ouroboros::Tensor<double> t2({3, 2});

for(size_t i = 0; i < t1.size(); i++) t1[i] = i + 1;
for(size_t i = 0; i < t2.size(); i++) t2[i] = i + 1;

auto multiply = [](double x, double y) { return x * y; };
auto outer = Ouroboros::outer<multiply>(t1, t2);
// Result shape: [2, 2, 3, 2]
// R(i0,i1,j0,j1) = t1(i0,i1) * t2(j0,j1) for all valid indices
```

### Broadcasting

```cpp
#include <ouroboros/tensor.hpp>

// Broadcasting allows operations on tensors with different but compatible shapes
Ouroboros::Tensor<double> t1({2, 3}, 1.0);
Ouroboros::Tensor<double> t2({2, 1}, 2.0);

auto divide = [](double x, double y) { return x / y; };
auto result = Ouroboros::broadcast<divide>(t1, t2);
// t2 is broadcast from [2, 1] to match [2, 3]
// Each row of t1 is divided by corresponding element in t2

// Broadcasting rules (similar to NumPy):
// - Dimensions are aligned from right to left
// - Size 1 dimensions are stretched
// - Missing dimensions are treated as 1
```

### Concatenation

```cpp
#include <ouroboros/tensor.hpp>

Ouroboros::Tensor<double> t1({2, 3}, 1.0);
Ouroboros::Tensor<double> t2({1, 3}, 2.0);
Ouroboros::Tensor<double> t3({3, 3}, 3.0);

// Concatenate along axis 0 (vertical stacking)
auto concat = Ouroboros::concat(0, t1, t2, t3);
// Result shape: [6, 3] (2+1+3 rows, 3 columns)

// All tensors must have same shape except along concatenation axis
```

### Flip and Transpose

```cpp
#include <ouroboros/tensor.hpp>

Ouroboros::Tensor<double> t({2, 3});
for(size_t i = 0; i < t.size(); i++) {
    t[i] = i + 1;  // [[1, 2, 3], [4, 5, 6]]
}

// Flip along axis 1 (reverse columns)
auto flipped = Ouroboros::flip(t, 1);  // [[3, 2, 1], [6, 5, 4]]

// Transpose (swap axes 0 and 1)
auto transposed = Ouroboros::transpose(t);  // Shape: [3, 2]
// [[1, 4], [2, 5], [3, 6]]

// Transpose arbitrary axes
Ouroboros::Tensor<double> t3d({2, 3, 4}, 1.0);
auto swapped = Ouroboros::transpose(t3d, 0, 2);  // Shape: [4, 3, 2]
```

### Boolean Tensor Operations

```cpp
#include <ouroboros/tensor.hpp>

Ouroboros::Tensor<bool> t1({2, 3}, true);
Ouroboros::Tensor<bool> t2({2, 3}, false);

// Element-wise logical operations
auto result1 = t1 && t2;  // All false
auto result2 = t1 || t2;  // All true
auto result3 = !t1;       // All false
auto result4 = t1 == t2;  // All false
auto result5 = t1 != t2;  // All true

// Operations with scalars
auto result6 = t1 && true;   // All true
auto result7 = t1 || false;  // All true
auto result8 = t1 == true;   // All true

// Scalar can be on either side
auto result9 = true && t1;   // Same as t1 && true
auto result10 = false || t1; // Same as t1 || false
```

### Complete Workflow Example

```cpp
#include <ouroboros/tensor.hpp>
#include <cmath>

int main() {
    // Create tensors
    Ouroboros::Tensor<double> t1({2, 3}, 1.0);
    Ouroboros::Tensor<double> t2({3, 2}, 2.0);
    
    // Matrix multiplication
    auto matmul_result = Ouroboros::matmul(t1, t2);  // [2, 2]
    
    // Apply transformation
    auto sin_func = [](double x) { return std::sin(x); };
    auto transformed = Ouroboros::transform<sin_func>(matmul_result);
    
    // Element-wise operations
    auto scaled = transformed * 2.0;
    auto shifted = scaled + 1.0;
    
    // Comparison
    auto mask = shifted > 0.5;  // Boolean tensor
    
    // Reduce along axis
    auto sum_reduce = [](Ouroboros::IdxIterator it, const double* data) {
        double sum = 0.0;
        for(auto i = it.begin(); i != it.end(); ++i) {
            sum += data[*i];
        }
        return sum;
    };
    auto sums = Ouroboros::reduce<sum_reduce>(0, shifted);
    
    std::cout << "Result: " << sums << std::endl;
    return 0;
}
```

## Performance Notes

1. **Memory Layout:** Row-major (C-style) ordering provides optimal cache locality for sequential access patterns.

2. **Stride-Based Indexing:** Pre-computed strides enable O(1) conversion from multi-dimensional to linear indices.

3. **Iterator Efficiency:** Iterators avoid repeated offset calculations by incrementing offsets directly.

4. **BLAS Integration:** Matrix operations (`matmul`, `matvecmul`) delegate to optimized BLAS routines for maximum performance.

5. **Move Semantics:** Move constructors and assignment operators avoid unnecessary copies of large tensors.

6. **Element-wise Operations:** Implemented with linear indexing to maximize cache hits and enable compiler auto-vectorization.

## Memory Management

1. **Ownership:** Tensors own their data arrays and free them in destructors using `delete[]`.

2. **Deep Copy:** Copy constructors and copy assignment perform deep copies of data arrays.

3. **Move Semantics:** Move operations transfer ownership without copying, leaving source objects in a valid but empty state.

4. **Shape Management:** Shape objects manage their own dimension and stride arrays independently.

5. **Pre-allocated Data:** Tensors can take ownership of pre-allocated arrays, but the arrays **must** be heap-allocated with `new[]`.

6. **Exception Safety:** Basic exception safety is provided, though the library generally assumes valid inputs.

## Notes

1. **Memory Management:** The library uses raw pointers and manual memory management. Tensors own their data and clean up in destructors.

2. **Type Requirements:** Most operations work with any numeric type `T`. Matrix operations (`matmul`, `matvecmul`) require `float` or `double` and use BLAS.

3. **Performance:** The library uses BLAS for matrix operations and includes optimization hints (`__always_inline`) for better performance.

4. **Error Handling:** The library generally assumes valid inputs. Shape mismatches or invalid indices may lead to undefined behavior.

5. **Comparison with NumPy:**
   - Like NumPy, Ouroboros provides N-dimensional arrays with element-wise operations
   - Broadcasting support through dedicated `broadcast()` function
   - Explicit function templates for custom operations (e.g., `transform`, `reduce`)
   - Memory layout is row-major (C-style), same as NumPy default
   - No dynamic typing - templates require compile-time type specification

6. **C++20 Features:** The library uses C++20 features including concepts and constexpr improvements.

## Best Practices and Common Patterns

### Choosing Between Compile-Time and Runtime Function Parameters

Many utility functions (`transform`, `reduce`, `accumulate`, `outer`, `broadcast`) support both compile-time and runtime function parameters:

**Use compile-time (template parameter) when:**
- Function is a simple lambda or function pointer
- Function doesn't capture variables
- Better performance is needed (enables more compiler optimizations)

```cpp
auto func = [](double x) { return x * x; };
auto result = Ouroboros::transform<func>(t);  // Preferred
```

**Use runtime (regular parameter) when:**
- Function captures variables from enclosing scope
- Function needs to be determined at runtime
- More flexibility is needed

```cpp
double factor = 10.0;
auto func = [factor](double x) { return x * factor; };  // Captures factor
auto result = Ouroboros::transform(func, t);  // Required
```

### Efficient Iteration Patterns

**For sequential access:**
```cpp
Ouroboros::Tensor<double> t({100, 100});
// Use linear indexing for best cache performance
for(size_t i = 0; i < t.size(); i++) {
    t[i] = compute(i);
}
```

**For dimension-aware access:**
```cpp
// Use IdxIterator with fixed dimensions
Ouroboros::IdxIterator iter(t.shape(), {{0, 5}});  // Fix first dimension to 5
for(auto it = iter.begin(); it != iter.end(); ++it) {
    auto idx = it.get_index();  // Get current multi-dimensional index
    t[*it] = compute(idx);      // *it gives linear offset
}
```

**For strided access:**
```cpp
// Use IdxIterator2 for slicing
Ouroboros::IdxIterator2 iter(
    t.shape(),
    {0, 0},      // Start
    {10, 10},    // End (exclusive)
    {2, 1}       // Step
);
for(auto it = iter.begin(); it != iter.end(); ++it) {
    process(t[*it]);
}
```

### Memory Management Best Practices

**Prefer move semantics for large tensors:**
```cpp
Ouroboros::Tensor<double> create_large_tensor() {
    Ouroboros::Tensor<double> t({1000, 1000}, 0.0);
    // ... fill tensor ...
    return t;  // Automatically uses move semantics
}

auto result = create_large_tensor();  // No copy, efficient
```

**Avoid unnecessary copies:**
```cpp
// Bad: unnecessary copy
void process(Ouroboros::Tensor<double> t) { /* ... */ }

// Good: pass by const reference if not modifying
void process(const Ouroboros::Tensor<double>& t) { /* ... */ }

// Good: pass by reference if modifying
void process(Ouroboros::Tensor<double>& t) { /* ... */ }

// Good: pass by value and move if consuming
void consume(Ouroboros::Tensor<double> t) {
    stored_tensor = std::move(t);
}
```

### Reshape vs. Creating New Tensor

**Reshape when:**
- Element count remains the same
- Memory layout doesn't need to change
- Want to reinterpret dimensions only

```cpp
Ouroboros::Tensor<double> t({2, 3, 4}, 1.0);
t.reshape({4, 6});  // Fast, just changes shape metadata
```

**Create new tensor when:**
- Need to reorder elements
- Want to keep original tensor unchanged
- Performing operations that change data

```cpp
auto transposed = Ouroboros::transpose(t);  // New tensor with reordered data
```

### Avoiding Common Pitfalls

**1. Pre-allocated data ownership:**
```cpp
// WRONG: Stack-allocated array
double data[6] = {1, 2, 3, 4, 5, 6};
Ouroboros::Tensor<double> t({2, 3}, data);  // Danger! Will try to delete[] stack memory

// CORRECT: Heap-allocated array
double* data = new double[6];
for(int i = 0; i < 6; i++) data[i] = i;
Ouroboros::Tensor<double> t({2, 3}, data);  // Safe, tensor takes ownership
```

**2. Using moved-from objects:**
```cpp
Ouroboros::Tensor<double> t1({2, 3}, 1.0);
auto t2 = std::move(t1);
// t1 is now in a valid but unspecified state
// DON'T use t1 after this point!
```

**3. Shape mismatch in operations:**
```cpp
Ouroboros::Tensor<double> t1({2, 3}, 1.0);
Ouroboros::Tensor<double> t2({3, 4}, 2.0);
// auto sum = t1 + t2;  // WRONG: shapes don't match

// Use broadcasting if shapes are compatible
auto result = Ouroboros::broadcast<add>(t1, t2);  // Check compatibility first
```

**4. Indexing out of bounds:**
```cpp
Ouroboros::Tensor<double> t({2, 3}, 1.0);
// t[{5, 10}] = 7.0;  // Undefined behavior! Out of bounds

// Always validate indices if uncertain
std::vector<size_t> idx = {1, 2};
if(idx[0] < t.shape()[0] && idx[1] < t.shape()[1]) {
    t[idx] = 7.0;  // Safe
}
```

### Optimization Tips

**1. Use BLAS operations when available:**
```cpp
// Prefer matmul for matrix multiplication
auto result = Ouroboros::matmul(a, b);  // Uses optimized BLAS

// Avoid manual element-wise multiplication for matrices
// (unless you specifically need Hadamard product)
```

**2. Minimize temporary allocations:**
```cpp
// Less efficient: multiple temporaries
auto result = (t1 + t2) * (t3 - t4) / t5;

// More efficient: reuse temporaries where possible
auto temp1 = t1 + t2;
auto temp2 = t3 - t4;
auto result = temp1 * temp2;
result /= t5;  // In-place division
```

**3. Use in-place operations when possible:**
```cpp
Ouroboros::Tensor<double> t({100, 100}, 1.0);

// Creates new tensor
auto result = t + 5.0;

// Modifies in-place (more efficient if you don't need original)
t += 5.0;
```

**4. Choose appropriate iteration strategy:**
```cpp
// For simple sequential operations, use linear indexing
for(size_t i = 0; i < t.size(); i++) {
    t[i] *= 2.0;  // Fast, cache-friendly
}

// For complex multi-dimensional logic, use iterators
Ouroboros::IdxIterator iter(t.shape());
for(auto it = iter.begin(); it != iter.end(); ++it) {
    auto idx = it.get_index();
    t[*it] = compute_based_on_position(idx);
}
```

### Integration with Standard Library

**Working with std::vector:**
```cpp
// Convert shape to vector
Ouroboros::Shape s({2, 3, 4});
std::vector<size_t> dims = s.to_vector();

// Use vector for indexing
std::vector<size_t> idx = {1, 2, 3};
double value = t[idx];
```

**Using standard algorithms:**
```cpp
Ouroboros::Tensor<double> t({10, 10}, 1.0);

// Get raw data pointer for standard algorithms
double* data = t.data();
const double* const_data = t.data();

// Use with standard library
std::fill(data, data + t.size(), 0.0);
auto max_elem = std::max_element(const_data, const_data + t.size());
std::transform(data, data + t.size(), data, [](double x) { return x * 2; });
```
