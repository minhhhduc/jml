# 🎯 Java ML Library (NumJa) - Complete Quick Reference Guide

This reference documents the full API of the closed-source, obfuscated production JAR distribution of NumJa. All dependencies are pre-compiled and bundled inside the `dist/` directory.

---

## 📊 QUICK MODULE REFERENCE

### 🧠 1. NumJa (Arrays & Dense Linear Algebra)
```java
import numja.core.NDArray;
import static numja.NumJa.*;

// Create arrays and matrices
NDArray x = array(new double[]{1.0, 2.0, 3.0});
NDArray zeros = zeros(10, 10);              // 10x10 zero matrix
NDArray ones = ones(5, 5);                  // 5x5 matrix of ones
NDArray linspace = linspace(0.0, 1.0, 100); // 100 values from 0.0 to 1.0

// Element-wise operations
NDArray y = add(x, x);
NDArray meanVal = mean(x);                  // Array mean value

// Multi-threaded operations & linear algebra
NDArray matMult = matmul(zeros, ones);
NDArray invMat = inv(ones);                 // Matrix inverse
```

### 🐼 2. Pandas (Data Loading & Manipulation)
```java
import pandas.Pandas;
import pandas.DataFrame;

// Load CSV file cascades searching local directories
DataFrame df = Pandas.read_csv("datasets/iris.csv");

// Indexing and Stats
System.out.println(df.head(5));
System.out.println(df.describe());
DataFrame subset = df.loc("target == 0");    // Query filtering
```

### 📈 3. Matplotlib & Seaborn (Visualizations)
```java
import matplotlib.Matplotlib;
import seaborn.Seaborn;
import numja.core.NDArray;

// Overloaded methods accept NDArray directly!
NDArray x = numja.NumJa.linspace(0, 10, 100);
NDArray y = numja.NumJa.array(new double[100]);

Matplotlib.clf();
Matplotlib.title("Smooth Curve Plot");
Matplotlib.plot(x, y, "Signal");
Matplotlib.scatter(x, y);
Matplotlib.show(); // Interactive Swing Window

// Smooth pixel grid imshow (No labels or number cells)
NDArray matrix2D = numja.NumJa.zeros(200, 200); // 200x200 pixel grid
Matplotlib.imshow(matrix2D);
Matplotlib.show(); 

// Seaborn Heatmap
int[][] cm = {{10, 0}, {1, 9}};
Seaborn.heatmap(cm, new String[]{"Class A", "Class B"});
Matplotlib.show();
```

### 🤖 4. SKLearn (Machine Learning Stack)

#### 🧼 Preprocessing & Scalers
```java
import sklearn.preprocessing.StandardScaler;
import sklearn.preprocessing.MinMaxScaler;
import sklearn.preprocessing.RobustScaler;

StandardScaler scaler = new StandardScaler();
NDArray X_scaled = scaler.fitTransform(X);
```

#### 📊 Datasets & Bunch Container
```java
import sklearn.datasets.Datasets;
import sklearn.datasets.Bunch;

// Load Iris dataset locally Cascade
Bunch iris = Datasets.loadIris();
NDArray X = iris.getData();
int[] y = iris.getTarget();

// Generate synthetic datasets
Bunch blobs = Datasets.makeBlobs(100, 2, 3, 42); // n_samples, n_features, centers, random_state
Bunch regData = Datasets.makeRegression(100, 5, 0.1, 42); // samples, features, noise, seed
```

#### 🤖 Classification (9 Models)
```java
import sklearn.ensemble.RandomForestClassifier;
import sklearn.neighbors.KNeighborsClassifier;
import sklearn.tree.DecisionTreeClassifier;
import sklearn.neural_network.MLPClassifier;

// Parallel Random Forest (Multi-threaded tree construction)
RandomForestClassifier rf = new RandomForestClassifier(50) // 50 trees
    .setMaxDepth(5)
    .setRandomState(42);
rf.fit(X_train, y_train);
int[] preds = rf.predict(X_test);
```

#### 📈 Regression (8 Models)
```java
import sklearn.linear_model.LinearRegression;
import sklearn.linear_model.Ridge;
import sklearn.linear_model.Lasso;
import sklearn.ensemble.RandomForestRegressor;

LinearRegression lr = new LinearRegression();
lr.fit(X_train, y_train);
double[] y_preds = lr.predict(X_test);
```

#### 🗄️ Model Selection, Tuning, and Pipelines
```java
import sklearn.pipeline.Pipeline;
import sklearn.model_selection.GridSearchCV;
import sklearn.model_selection.ModelSelection;

// 1. Train-test split
Object[] split = ModelSelection.trainTestSplit(X, y, 0.3, 42);
NDArray X_train = (NDArray) split[0];
NDArray X_test = (NDArray) split[1];
int[] y_train = (int[]) split[2];

// 2. Machine Learning Pipeline (Imputer -> Scaler -> Model)
Pipeline pipe = new Pipeline()
    .add(new sklearn.preprocessing.SimpleImputer("mean"))
    .add(new sklearn.preprocessing.StandardScaler())
    .add(rf);

// 3. Multi-threaded Grid Search CV
GridSearchCV grid = new GridSearchCV(pipe, paramGrid)
    .setCV(5)
    .setNJobs(-1); // Parallel threads
grid.fit(X_train, y_train);
```

---

## 🛠️ DISTRIBUTION & VERIFICATION (`dist/` folder)

The distribution folder `dist/` is pre-obfuscated and standalone:

```text
dist/
├── numja.jar          (Core array mathematics)
├── pandas.jar         (DataFrame table parsing)
├── matplotlib.jar     (Visual plotting & imshow)
├── seaborn.jar        (Heatmap visualizations)
├── sklearn.jar        (Machine Learning models)
└── libs/              (Pre-compiled third-party JARs)
```

### ☕ Compiling and Running Client Code:
You can compile and run any client application referencing the `dist/` jars without Maven:

```powershell
# Compile
javac -cp "dist/*;dist/libs/*;." MyMLApp.java

# Run
java -ea -cp "dist/*;dist/libs/*;." MyMLApp
```
*(Option `-ea` enables Java assertions to verify pipeline constraints).*
