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
Provides machine learning estimators, datasets, and preprocessing scalers in Java, matching Python's Scikit-Learn API exactly. Every estimator uses the familiar, standardized Python method signatures (`fit()`, `predict()`, `score()`, `transform()`, and `fitTransform()`) to deliver a fully equivalent developer experience.

> [!NOTE]
> The API and operational paradigms of the `sklearn` module strictly duplicate the Scikit-Learn Python library structure. For academic and technical reference, see the foundational publication:  
> Pedregosa et al., "Scikit-learn: Machine Learning in Python", *Journal of Machine Learning Research*, 12, pp. 2825-2830, 2011.

### 4.1 Supervised Classifiers & Regressors
Encapsulates supervised learning algorithms for qualitative classification and quantitative regression.

* **Classifiers (`sklearn.ensemble.RandomForestClassifier`, `sklearn.tree.DecisionTreeClassifier`, `sklearn.neighbors.KNeighborsClassifier`, `sklearn.linear_model.LogisticRegression`, `sklearn.svm.SVC`, `sklearn.naive_bayes.GaussianNB`, `sklearn.ensemble.GradientBoostingClassifier`, `sklearn.ensemble.AdaBoostClassifier`, `sklearn.neural_network.MLPClassifier`):**
  * `fit(NDArray X, int[] y)`: Trains the classification model on features $X \in \mathbb{R}^{m \times n}$ and label targets $y \in \mathbb{Z}^m$.
  * `predict(NDArray X)`: Generates predicted labels $y_{pred} \in \mathbb{Z}^m$ for test features $X$.
  * `score(NDArray X, int[] y)`: Computes the classification accuracy on test features $X$ and targets $y$.

* **Regressors (`sklearn.linear_model.LinearRegression`, `sklearn.linear_model.Ridge`, `sklearn.linear_model.Lasso`, `sklearn.tree.DecisionTreeRegressor`, `sklearn.ensemble.RandomForestRegressor`, `sklearn.neighbors.KNeighborsRegressor`, `sklearn.svm.SVR`, `sklearn.neural_network.MLPRegressor`):**
  * `fit(NDArray X, double[] y)`: Trains the regression model on features $X \in \mathbb{R}^{m \times n}$ and continuous targets $y \in \mathbb{R}^m$.
  * `predict(NDArray X)`: Generates continuous predictions $y_{pred} \in \mathbb{R}^m$ for test features $X$.
  * `score(NDArray X, double[] y)`: Computes the $R^2$ coefficient of determination on test features $X$ and targets $y$.

### 4.2 Numerical Transformers & Preprocessing
Enables standard feature scaling and data imputation transforms.

* **Scalers & Imputers (`sklearn.preprocessing.StandardScaler`, `sklearn.preprocessing.MinMaxScaler`, `sklearn.preprocessing.RobustScaler`, `sklearn.impute.SimpleImputer`):**
  * `fit(NDArray X)`: Computes the transformation statistics (e.g. column-wise mean $\mu$ and standard deviation $\sigma$ for standardization).
  * `transform(NDArray X)`: Standardizes or scales the features $X$ using fitted statistics.
  * `fitTransform(NDArray X)`: Fits to features and returns the transformed/scaled NDArray.
* **Categorical & Feature Transformers (`sklearn.preprocessing.LabelEncoder`, `sklearn.preprocessing.OneHotEncoder`, `sklearn.preprocessing.PolynomialFeatures`):**
  * `fitTransform(NDArray X)`: Encodes categorical labels or synthesizes higher-order polynomial interaction features.

### 4.3 Model Selection & Validation: `sklearn.model_selection.ModelSelection`
Facilitates partitioning of datasets and testing of generalization capacities.
* `trainTestSplit(NDArray X, int[] y, double test_size, int seed)`: Randomly splits features and targets into a partitioned array structure `Object[] {X_train, X_test, y_train, y_test}`.
* `crossValScore(Object estimator, NDArray X, int[] y, int cv)`: Performs multi-threaded $K$-fold cross-validation splits and returns an array of validation scores.

### 4.4 Dataset Utilities & Bunch Container
Provides local datasets and data/target wrapping envelopes.

* **Dataset Loaders (`sklearn.datasets.Datasets`):**
  * `loadIris()`: Returns a `Bunch` container containing the Iris classification dataset.
  * `makeRegression(int samples, int features, double noise, int seed)`: Generates synthetic regression datasets.
  * `makeBlobs(int samples, int features, int centers, int seed)`: Generates synthetic clustering datasets.
* **Bunch Envelope (`sklearn.datasets.Bunch`):**
  * `getData()`: Returns the features NDArray.
  * `getTarget()`: Returns the target labels array (e.g. `int[]` or `double[]`).
