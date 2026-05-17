# NumJa: Scientific Technical Specification & API Reference
> **Release Version 0.2.0 - Closed-Source Release Architecture**
> This document serves as the formal mathematical specification and scientific API catalog for the **NumJa** modular ecosystem. NumJa provides a highly efficient, standalone Java runtime simulating Python's scientific computation stack (NumPy, Pandas, Matplotlib, Seaborn, and Scikit-Learn) with built-in multi-threaded accelerations.

---

## 1. NumJa Core: Dense Vector & Matrix Algebra
The foundational mathematical engine for dense vector and matrix representations, vector operations, and matrix factorizations.

### 1.1 Dense Array Class: `numja.core.NDArray`
A dense array wrapping EJML's row-major dense matrix (`DMatrixRMaj`), representing a 1D column vector ($A \in \mathbb{R}^{d_1}$) or a 2D matrix ($A \in \mathbb{R}^{d_1 \times d_2}$).
* **Algebraic Methods & Transformations:**
  * $\operatorname{shape}(A)$: Returns the dimensional tuple $(d_1)$ for 1D or $(d_1, d_2)$ for 2D.
  * $\operatorname{reshape}(A, s_1, s_2)$: Re-indexes elements into a new 1D or 2D shape under constraint $\prod d_i = \prod s_j$.
  * $A^T$: Matrix transposition ($A^T_{ij} = A_{ji}$).
  * `toArray()`: Serializes data into a native Java 2D array (`double[][]`).
  * `toDoubleArray()`: Serializes data into a flat 1D Java array (`double[]`).
* **Vectorized Algebra & Operations:**
  * `add(NDArray)` / `add(double)`: Computes matrix-matrix addition ($C = A + B$) or element-wise scalar addition.
  * `subtract(NDArray)` / `subtract(double)`: Computes matrix-matrix subtraction ($C = A - B$) or element-wise scalar subtraction.
  * `multiply(NDArray)` / `multiply(double)`: Computes Hadamard (element-wise) product ($C = A \odot B$) or scalar multiplication.
  * `divide(NDArray)` / `divide(double)`: Computes element-wise division ($C_i = A_i / B_i$) or scalar division.
  * `dot(NDArray)` / `matmul(NDArray)`: General Matrix Multiplication (GEMM) using high-performance parallel BLAS-like routines:
    $$C_{ik} = \sum_{j=1}^{m} A_{ij} B_{jk}$$

### 1.2 Factory & Statistical Core: `numja.NumJa`
Exposes statistical operators and array instantiation kernels:
* `array(double[][])` / `array(double...)`: Creates `NDArray` representing a 1D column vector or a 2D matrix.
* `zeros(int...)` / `ones(int...)`: Instantiates zero or identity vectors/matrices of target dimensions.
* `linspace(a, b, n)`: Computes an equidistant partition of the closed interval $[a, b]$ with cardinality $n$.
* $\mu(A)$ / $\sigma(A)$: Computes the empirical mean and standard deviation of the matrix:
  $$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i, \quad \sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_i - \mu)^2}$$

### 1.3 Matrix Decompositions: `numja.linalg.LinAlg`
Rigorous numeric implementations of core linear algebra routines:
* $A^{-1}$: Matrix inversion via LU decomposition with partial pivoting.
* $\det(A)$: Determinant of a square matrix $A \in \mathbb{R}^{n \times n}$.
* $Ax = b$: Solver for linear systems of equations using QR factorization or Gaussian elimination.
* $\operatorname{Tr}(A)$: Matrix trace computed as $\sum_{i=1}^n A_{ii}$.
* **Singular Value Decomposition (SVD):**
  $$A = U \Sigma V^T$$
  Decomposes $A \in \mathbb{R}^{m \times n}$ into orthogonal matrices $U, V$ and singular value matrix $\Sigma$.
* **QR Decomposition:**
  $$A = Q R$$
  Factorizes square/rectangular matrix $A$ into orthogonal matrix $Q$ and upper triangular matrix $R$.
* **Eigendecomposition:**
  $$A v = \lambda v$$
  Extracts eigenvalues $\lambda_i \in \mathbb{C}$ and eigenvectors $v_i \in \mathbb{C}^n$ for diagonalizable matrices.
* **Ordinary Least Squares (OLS):**
  $$\min_{x} \|Ax - b\|_2$$
  Solves the overdetermined linear system using pseudoinverse or QR techniques.

---

## 2. Pandas: Relational & Tabular Data Structures
A rigorous framework for structural manipulation of tabular data representations under labeled coordinate systems.

### 2.1 Tabular Loader: `pandas.Pandas`
Facilitates robust serialization/deserialization:
* `read_csv(path, options)`: Formally parses comma-separated relations into tabular data sheets. Supports schema options including custom delimiters (separator $\delta$), header row selection, skipped rows, and primary index column definitions.
* `read_json(path)`: Deserializes structural JSON documents into relations.

### 2.2 Relational Manifold: `pandas.DataFrame`
A 2D structural table with vertical and horizontal indexing maps.
* **Structural Properties:**
  * $\operatorname{shape}(DF)$: Dimension map $\mathcal{M}_{m \times n}$ representing $m$ observations across $n$ features.
  * `head(k)` / `tail(k)`: Samples the top/bottom $k$ rows.
  * `columns()`: Retransports the column header list.
  * `describe()`: Generates a complete mathematical statistical profile:
    $$\{\text{count}, \mu, \sigma, x_{\min}, x_{0.25}, x_{0.50}, x_{0.75}, x_{\max}\}$$
* **Slicing & Algebraic Projections:**
  * `loc` (via `DataFrameLoc`): Projection mapping via string labels.
  * `iloc` (via `DataFrameILoc`): Coordinate-based integer slicing.
  * `drop(columns)` / `drop_duplicates()`: Removes features or row-wise duplicates.
  * `dropna()` / `fillna(value)`: Cleans incomplete manifolds by dropping or imputing missing elements (NaN).
  * `sort_values(col)`: Performs ordinal sort based on target column values.
  * `groupby(col)`: Performs partition clustering based on feature equivalence classes.
  * `toNDArray()`: Casts the numeric values of the DataFrame into a dense matrix $A \in \mathbb{R}^{m \times n}$.

### 2.3 Labeled Vector: `pandas.Series`
A 1D labeled array representing a single random variable (column vector) with descriptive univariate statistical methods.

---

## 3. Matplotlib & Seaborn: Visual Data Representation
Two-dimensional graphical visualization engines utilizing high-performance pixel-level rendering.

### 3.1 Graphical Engine: `matplotlib.Matplotlib`
* `plot(x, y)`: Renders piecewise linear continuous approximations of functions.
* `scatter(x, y)`: Plots discrete experimental distributions $S = \{(x_i, y_i)\}$.
* `title(str)` / `xlabel(str)` / `ylabel(str)`: Sets textual descriptors on Cartesian projections.
* `clf()`: Clears the graphic state.
* `show()`: Displays an interactive GUI frame showcasing the chart.
* `savefig(path)`: Exports the vector graphic to a rasterized image file (PNG).
* `imshow(NDArray)`: Renders 2D grid intensities as a continuous thermal grid, automatically scaling pixel intensities and displaying a continuous color-bar mapping.

### 3.2 Statistical Visuals: `seaborn.Seaborn`
* `heatmap(matrix, classes)`: Standardized visualization of Confusion Matrices or covariance matrices, incorporating smooth gradient color ramps and labeled coordinate bounds.

---

## 4. SKLearn: Machine Learning Module
A fully integrated, multi-threaded numerical estimation framework conforming to the Python Scikit-Learn paradigm.

### 4.1 Supervised Classifiers
Supervised estimators for predicting qualitative targets $y \in \{C_1, C_2, \dots, C_k\}$:
* `sklearn.tree.DecisionTreeClassifier`: Minimizes classification impurity metrics (Gini index or Information Entropy):
  $$I_{Gini}(p) = 1 - \sum_{i=1}^{k} p_i^2, \quad I_{Entropy}(p) = -\sum_{i=1}^{k} p_i \log_2(p_i)$$
* `sklearn.ensemble.RandomForestClassifier`: Ensembles $B$ distinct decision trees. Features parallel training using CPU multi-threading via `n_jobs=-1`:
  $$\hat{y} = \operatorname{mode}\{\hat{y}_1, \hat{y}_2, \dots, \hat{y}_B\}$$
* `sklearn.neighbors.KNeighborsClassifier`: Non-parametric classification based on Minkowski distances in metric spaces $\mathbb{R}^d$:
  $$d(p, q) = \left( \sum_{i=1}^d |p_i - q_i|^r \right)^{1/r}$$
* `sklearn.linear_model.LogisticRegression`: Optimizes cross-entropy loss via gradient descent with Sigmoid mapping:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
* `sklearn.svm.SVC`: Computes optimal hyperplanes maximizing margin bounds $\frac{2}{\|w\|}$ in high-dimensional kernel spaces.
* `sklearn.naive_bayes.GaussianNB`: Classification based on Bayes' theorem assuming feature independence:
  $$P(X_i | y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left( -\frac{(X_i - \mu_y)^2}{2\sigma_y^2} \right)$$
* `sklearn.ensemble.GradientBoostingClassifier` / `AdaBoostClassifier`: Boosted weak learners optimized through sequential cost reduction.
* `sklearn.neural_network.MLPClassifier`: Multi-Layer Perceptron trained using backpropagation with custom activations (Logistic, ReLU, Tanh) and categorical Softmax outputs.

### 4.2 Supervised Regressors
Predicts quantitative target vectors $y \in \mathbb{R}$:
* `sklearn.linear_model.LinearRegression`: Fits parameters $\theta$ to minimize Residual Sum of Squares (RSS):
  $$J(\theta) = \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$
* `sklearn.linear_model.Ridge`: Adds $L_2$ regularization penalty to prevent parameter inflation:
  $$J(\theta) = \text{RSS} + \alpha \sum_{j=1}^n \theta_j^2$$
* `sklearn.linear_model.Lasso`: Adds $L_1$ regularization penalty promoting feature sparsity:
  $$J(\theta) = \text{RSS} + \alpha \sum_{j=1}^n |\theta_j|$$
* `sklearn.tree.DecisionTreeRegressor` / `RandomForestRegressor`: Tree-based piecewise continuous regression estimators.
* `sklearn.neighbors.KNeighborsRegressor` / `SVR` / `MLPRegressor`: Regressors utilizing local geometric density, Support Vector formulations, and Multi-Layer Perceptron architectures.

### 4.3 Numerical Feature Transforms (Preprocessing)
* `sklearn.preprocessing.StandardScaler`: Transforms features to conform to the standard normal distribution $\mathcal{N}(0, 1)$:
  $$z = \frac{x - \mu}{\sigma}$$
* `sklearn.preprocessing.MinMaxScaler`: Scales bounded domains linearly to target interval $[a, b]$:
  $$x' = a + \frac{(x - x_{\min})(b - a)}{x_{\max} - x_{\min}}$$
* `sklearn.preprocessing.RobustScaler`: Scales features using median $\tilde{x}$ and Interquartile Range (IQR) to secure resistance against outliers:
  $$x' = \frac{x - \tilde{x}}{\text{IQR}}$$
* `sklearn.preprocessing.LabelEncoder` / `OneHotEncoder`: Maps categorical values to numeric keys or binary sparse vector spaces.
* `sklearn.preprocessing.PolynomialFeatures`: Synthesizes higher-order interaction features:
  $$\phi(x) = [1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]$$
* `sklearn.impute.SimpleImputer`: Fills incomplete records using statistical estimators (Mean, Median, or Constant value).

### 4.4 Model Selection & Pipelines
* `sklearn.model_selection.ModelSelection`:
  * `trainTestSplit()`: Partitions datasets randomly while preserving relative label frequencies.
  * `crossValScore()`: Computes $K$-fold cross-validation scores to assess generalization.
* `sklearn.model_selection.GridSearchCV`: Performs brute-force multi-threaded parameter sweeps (`n_jobs=-1`) to identify optimal estimators.
* `sklearn.pipeline.Pipeline`: Implements a unified composite pipeline structure wrapping sequential transformations and estimation steps.

### 4.5 Datasets & Utilities
* `sklearn.datasets.Datasets`: High-level loaders (`loadIris()`) and synthetic dataset generators (`makeRegression()`, `makeBlobs()`).
* `sklearn.datasets.Bunch`: A key-value dictionary container for data and targets.
