# Numerical Analysis Coursework

This directory contains programming assignments from my Numerical Analysis course. This course focused on the design, analysis, and implementation of algorithms for solving continuous mathematical problems, with an emphasis on accuracy, stability, and computational efficiency.

## Key Learnings from the Course

* **Root-Finding Algorithms:** Implemented iterative methods for finding roots of non-linear equations.
* **Numerical Integration:** Explored techniques for approximating definite integrals.
* **Numerical Differentiation:** Implemented methods for approximating derivatives.
* **Solving Linear Systems:** Applied iterative and direct methods for solving systems of linear equations.
* **Eigenvalue Problems:** Learned about methods for computing eigenvalues and eigenvectors of matrices.
* **Approximation Theory:** Understood polynomial interpolation and approximation techniques.
* **Error Analysis:** Critically evaluated the sources and propagation of errors (truncation, round-off) in numerical methods.

## Assignments

The assignments are listed from the latest (most advanced) to the earliest, reflecting the progression of concepts covered in the course.

### 1. Homework 12: Jacobi and Power Methods

**Description:** This assignment involved implementing two iterative methods for linear algebra problems: the Jacobi Method for solving systems of linear equations (<span class="math-inline">Ax\=b</span>) and the Power Method for finding the dominant eigenvalue and its corresponding eigenvector of a matrix.
**Advanced Aspects:** Understanding iterative convergence, spectral properties of matrices (dominant eigenvalue), and the numerical stability of iterative solvers for large systems.
**[View Code](./HW12-PowerMethod/NA_HW12_Jacobi_Power_Method.ipynb)**

### 2. Homework 11: Numerical Differentiation

**Description:** Focused on approximating derivatives of functions using various finite difference formulas. This included implementing methods like the two-point forward difference formula and the three-point central difference formula.
**Advanced Aspects:** Understanding the derivation of finite difference approximations, analyzing truncation error, and the impact of step size (`h`) on accuracy.
**[View Code](./HW11-NumDiff/NA_HW11_Num_Diff.ipynb)**

### 3. Homework 8: Gaussian Elimination

**Description:** Implemented **Gaussian Elimination** for solving square systems of linear equations (<span class="math-inline">Ax\=b</span>). This included implementing both the forward elimination and back-substitution steps.
**Advanced Aspects:** Direct methods for linear systems, understanding row operations, and the computational cost of direct solvers.
**[View Code](./HW8-GaussElim/NA_HW8_Gauss_Elim.ipynb)**

### 4. Homework 7: Euler's Method for ODEs

**Description:** Implemented **Euler's Method**, a fundamental numerical technique for approximating solutions to ordinary differential equations (ODEs) given an initial condition and a step size.
**Advanced Aspects:** Understanding discrete approximations of continuous processes, analyzing local and global truncation errors, and visualizing solution trajectories.
**[View Code](./HW7-EulersMethod/NA_HW7_Eulers_ODE.ipynb)**

### 5. Homework 6: Simpson's Rule for Integration

**Description:** Implemented **Simpson's Rule** for numerical integration. This assignment focused on applying this higher-order quadrature rule to approximate definite integrals of various functions.
**Advanced Aspects:** Understanding weighted sums for integration, improved accuracy compared to simpler methods (like Midpoint/Trapezoidal), and error analysis for composite rules.
**[View Code](./HW6-SimpsonsRule/NA_HW6_Simpsons_Int.ipynb)**

### 6. Homework 5: Midpoint and Trapezoidal Rules for Integration

**Description:** This assignment introduced fundamental numerical integration techniques: the **Midpoint Rule** and the **Trapezoidal Rule**. It involved implementing these methods to approximate definite integrals.
**Advanced Aspects:** Understanding the geometric interpretation of these rules, basic concepts of numerical quadrature, and initial analysis of approximation errors.
**[View Code](./HW5-MidTrap-Int/NA_HW5_MidTrap_Int.ipynb)**

### 7. Homework 1: Vector and Matrix Operations

**Description:** This introductory assignment covered fundamental vector and matrix operations in Python, including addition, scalar multiplication, dot product, and matrix multiplication.
**Advanced Aspects:** Implementing linear algebra operations from first principles, understanding the rules of matrix arithmetic, and building foundational numerical computing skills.
**[View Code](./HW1-VectorMatrix/NA_HW1_VectorMatrix.ipynb)**

## C++ Snippets (Related to Numerical Analysis)

This folder also contains smaller C++ code snippets from various explorations, demonstrating core numerical concepts in C++.

* **[Matrix-Vector Multiplication Snippet](./snippets/NA_Snippet_MatVec_Mult.ipynb)**
* **[Vector Class Snippet](./snippets/NA_Snippet_Vector_Class.ipynb)**
* **[Matrix Multiplication Snippet](./snippets/NA_Snippet_Matrix_Mult.ipynb)**