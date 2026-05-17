# NumJa: High-Performance Matrix Algebra & Statistical Learning Framework for Java
> **Release Version 0.2.0 - Closed-Source Modular Release**
> NumJa is a high-performance, zero-dependency, and mathematically rigorous matrix algebra and statistical learning library developed natively for the Java Virtual Machine (JVM). It offers an intuitive, Python-equivalent syntax simulating NumPy, Pandas, Matplotlib, Seaborn, and Scikit-Learn, optimized for low-latency offline executions.

---

## Scientific Abstract & Motivation
In scientific computing and data science, the Python ecosystem (NumPy, SciPy, Scikit-Learn) has established dominance due to expressive API semantics and vectorized execution. However, deploying these models onto the JVM often introduces significant JNI overhead, dependency conflicts, or deployment complexities. 

**NumJa** bridges this gap by introducing a native, fully modular, and highly optimized mathematical framework. By wrapping underlying multi-threaded BLAS-like routines, NumJa enables high-throughput matrix-oriented data pipelines directly within the JVM. Furthermore, the compiled modules are protected via advanced ProGuard Obfuscation to secure proprietary algorithmic assets during offline deployment.

---

## Architectural Specifications & Decoupling

NumJa is distributed as a suite of five decoupled, highly interoperable JAR modules targeting Java 11 through Java 25+:

```text
dist/
├── matplotlib.jar
├── numja.jar
├── pandas.jar
├── seaborn.jar
├── sklearn.jar
├── libs/
│   ├── commons-math3-3.6.1.jar
│   ├── ejml-core-0.43.1.jar
│   ├── ejml-ddense-0.43.1.jar
│   └── jfreechart-1.5.3.jar
└── datasets/
    ├── breast_cancer.csv
    ├── california_housing.csv
    ├── diabetes.csv
    ├── diamonds.csv
    ├── digits.csv
    ├── flights.csv
    ├── iris.csv
    ├── penguins.csv
    ├── tips.csv
    ├── titanic.csv
    └── wine.csv
```

> [!NOTE]  
> **Closed-Source Release Paradigm:**  
> The underlying implementation source code in `modules/` is mathematically modularized and secured under strict `.gitignore` configurations. Consumers only require the compiled, highly optimized binaries in `dist/` and standard Java dependencies, providing a zero-install, portable offline execution model.

---

## Core Scientific Modules & Mathematical Formulations

### 1. NumJa Core: Dense Vector & Matrix Algebra
Handles 1D vector and 2D matrix representations and rigorous numerical linear algebra.

* **Vector & Matrix Concept:**
  The central class `numja.core.NDArray` wraps EJML's `DMatrixRMaj` to encapsulate dense 1D column vectors ($A \in \mathbb{R}^{d_1}$) and 2D matrices ($A \in \mathbb{R}^{d_1 \times d_2}$), supporting reshaping, transposition, and vectorized operations.
* **General Matrix Multiplication (GEMM):**
  Matrix multiplication on 2D arrays is mapped via parallelized, cache-friendly implementations:
  $$C_{ik} = \sum_{j=1}^{m} A_{ij} B_{jk}$$
* **Linear Algebra Decompositions (`numja.linalg.LinAlg`):**
  Provides accurate implementations of essential numerical algorithms:
  * **Matrix Inverse ($A^{-1}$):** Solved via LU factorization with partial pivoting.
  * **System Solver ($Ax = b$):** Resolves exact linear systems using QR decomposition or Gaussian elimination.
  * **Singular Value Decomposition (SVD):**
    $$A = U \Sigma V^T$$
  * **QR Decomposition:**
    $$A = Q R$$
  * **Eigendecomposition:**
    $$A v = \lambda v$$
  * **Ordinary Least Squares (OLS):**
    $$\min_{x} \|Ax - b\|_2$$

---

### 2. Pandas: Relational Tabular Analysis
Manages structural data representations under discrete labeled indexing coordinate systems.

* **Laminated Data Structures:**
  Provides the labeled 1D vector `Series` and the 2D tabular matrix `DataFrame`.
* **Descriptive Structural Profiling:**
  The `describe()` routine computes empirical distributions across features:
  $$\{\text{count}, \mu, \sigma, x_{\min}, x_{0.25}, x_{0.50}, x_{0.75}, x_{\max}\}$$
* **Manifold Mappings:**
  * **`loc` (Label Projection):** Coordinate slicing via explicit column/row header tags.
  * **`iloc` (Coordinate Slicing):** Integer-based coordinate indexing projections.
  * **I/O Pipelines:** High-performance, parameterizable `read_csv` and `read_json` parsers supporting separator delimiter $\delta$, custom headers, and indexing keys.

---

### 3. Matplotlib & Seaborn: Two-Dimensional Graphical Visualizations
Implements high-fidelity Cartesian plotting and pixel-level grid rendering.

* **State-Machine Plotting (`Matplotlib`):**
  Manages continuous function mappings (`plot`), discrete scattering distributions (`scatter`), and graphical canvas manipulation (`clf()`, `title()`, `savefig()`).
* **High-Performance Image Grid Rendering (`imshow`):**
  Supports direct intensity mappings of $A \in \mathbb{R}^{m \times n}$ to continuous thermal pixel grid projections, integrating automatic color-bar density bars.
* **Statistical Distribution Mapping (`Seaborn`):**
  Renders labeled confusion matrices and covariance mappings using multi-tonal gradient ramps (`heatmap`).

---

### 4. SKLearn: Machine Learning Module
Provides machine learning estimators, datasets, and preprocessing scalers in Java, matching Python's Scikit-Learn API exactly. Every estimator replicates Python's method signatures (`fit()`, `predict()`, `score()`, `transform()`, and `fitTransform()`) to ensure a fully equivalent user experience.

> [!NOTE]
> The API and design of the `sklearn` module strictly replicate the Scikit-Learn Python library structure. For academic and technical reference, see:  
> Pedregosa et al., "Scikit-learn: Machine Learning in Python", *Journal of Machine Learning Research*, 12, pp. 2825-2830, 2011.

```java
import sklearn.datasets.Datasets;
import sklearn.datasets.Bunch;
import sklearn.preprocessing.StandardScaler;
import sklearn.model_selection.ModelSelection;
import sklearn.ensemble.RandomForestClassifier;
import numja.core.NDArray;

public class MLExample {
    public static void main(String[] args) {
        // 1. Load the Iris dataset
        Bunch iris = Datasets.loadIris();
        NDArray X = iris.getData();
        int[] y = iris.getTarget();

        // 2. Partition into training and testing splits
        Object[] split = ModelSelection.trainTestSplit(X, y, 0.3, 42);
        NDArray X_train = (NDArray) split[0];
        NDArray X_test = (NDArray) split[1];
        int[] y_train = (int[]) split[2];
        int[] y_test = (int[]) split[3];

        // 3. Scale features using Z-score standardization
        StandardScaler scaler = new StandardScaler();
        NDArray X_train_scaled = scaler.fitTransform(X_train);
        NDArray X_test_scaled = scaler.transform(X_test);

        // 4. Instantiate and fit a multi-threaded Random Forest Classifier
        RandomForestClassifier rf = new RandomForestClassifier(100) // 100 trees
            .setMaxDepth(5)
            .setRandomState(42);
        rf.fit(X_train_scaled, y_train);

        // 5. Predict and evaluate performance
        int[] predictions = rf.predict(X_test_scaled);
        double accuracy = rf.score(X_test_scaled, y_test);
        System.out.println("Model Test Accuracy: " + accuracy);
    }
}
```

---
**NumJa — Bridging JVM performance with the expressive elegance of mathematical Python.**
