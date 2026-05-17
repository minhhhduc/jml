# SKLearn Module - Machine Learning Library

Complete machine learning module for Java with scikit-learn style API.

## Features

### 1. **Clustering** (`sklearn.cluster`)

#### KMeans
```java
KMeans kmeans = new KMeans(n_clusters);
kmeans.fit(X);
int[] labels = kmeans.predict(X_new);
```

#### Gaussian Mixture Model (GMM)
```java
GaussianMixture gmm = new GaussianMixture(n_components);
gmm.fit(X);
int[] labels = gmm.predict(X);
double[] weights = gmm.getWeights();
```

### 2. **Preprocessing** (`sklearn.preprocessing`)

#### StandardScaler
Standardize features (mean=0, std=1):
```java
StandardScaler scaler = new StandardScaler();
NDArray X_scaled = scaler.fitTransform(X);
NDArray X_original = scaler.inverseTransform(X_scaled);
```

### 3. **Neural Networks** (`sklearn.neural_network`)

#### Activation Functions
```java
// Supported: sigmoid, relu, tanh, softmax
double result = Activations.sigmoid(x);
double[] probs = Activations.softmax(logits);
```

#### MLPClassifier
Multi-layer perceptron for classification:
```java
MLPClassifier mlp = new MLPClassifier(64, 32)  // Hidden layers
    .setActivation("relu")
    .setLearningRate(0.01)
    .setMaxIter(200);

mlp.fit(X_train, y_train);
int[] predictions = mlp.predict(X_test);
double[][] probabilities = mlp.predictProba(X_test);
```

#### Autoencoder
Unsupervised feature learning and dimensionality reduction:
```java
Autoencoder ae = new Autoencoder(input_dim, encoding_dim);
ae.fit(X);

// Encode to latent space
NDArray encoded = ae.transform(X);

// Reconstruct from latent
NDArray reconstructed = ae.inverseTransform(encoded);
```

### 4. **Dimensionality Reduction** (`sklearn.decomposition`)

#### PCA (Principal Component Analysis)
```java
PCA pca = new PCA(n_components);
NDArray X_reduced = pca.fitTransform(X);

// Inverse transform (approximate reconstruction)
NDArray X_reconstructed = pca.inverseTransform(X_reduced);

// Get explained variance
double[] variance_ratio = pca.getExplainedVarianceRatio();
```

## Usage Examples

### Example 1: Clustering with KMeans + StandardScaler
```java
// Prepare data
NDArray X = NumJa.array(data);

// Standardize
StandardScaler scaler = new StandardScaler();
NDArray X_scaled = scaler.fitTransform(X);

// Cluster
KMeans kmeans = new KMeans(3);
kmeans.fit(X_scaled);
int[] labels = kmeans.getLabels();
```

### Example 2: Dimensionality Reduction with PCA
```java
// Original high-dimensional data
NDArray X = NumJa.array(high_dim_data);

// Reduce to 2D for visualization
PCA pca = new PCA(2);
NDArray X_2d = pca.fitTransform(X);

// Keep 95% of variance
PCA pca95 = new PCA(50);  // Adjust based on variance ratio
NDArray X_reduced = pca95.fitTransform(X);
```

### Example 3: Neural Network Classification
```java
// Prepare data
double[][] X_train = ...;
int[] y_train = ...;

// Create network
MLPClassifier nn = new MLPClassifier(128, 64, 32)
    .setActivation("relu")
    .setLearningRate(0.001)
    .setMaxIter(1000);

// Train
nn.fit(NumJa.array(X_train), y_train);

// Predict
double[][] probabilities = nn.predictProba(X_test);
int[] predictions = nn.predict(X_test);
```

### Example 4: Autoencoder for Feature Extraction
```java
// Train autoencoder to learn compressed representation
Autoencoder ae = new Autoencoder(784, 128, 256);  // MNIST-like
ae.fit(X_images);

// Extract features (encoded representation)
NDArray features = ae.transform(X_images);

// Use these features for downstream tasks
```

## API Reference

### Clustering
- `KMeans(n_clusters)` - K-means clustering
  - `fit(X)` - Fit to data
  - `predict(X)` - Predict clusters
  - `getCentroids()` - Get cluster centers

- `GaussianMixture(n_components)` - Probabilistic clustering
  - `fit(X)` - Fit EM algorithm
  - `predict(X)` - Predict clusters
  - `getWeights()` - Get mixing coefficients

### Preprocessing
- `StandardScaler()` - Feature scaling
  - `fit(X)` - Compute mean and std
  - `transform(X)` - Apply scaling
  - `fitTransform(X)` - Fit and transform
  - `inverseTransform(X)` - Reverse scaling

### Neural Networks
- `Activations` - Static activation functions
  - `sigmoid(x)`, `relu(x)`, `tanh(x)`, `softmax(x)`

- `MLPClassifier(hidden_sizes...)` - Multi-layer perceptron
  - `fit(X, y)` - Train on labeled data
  - `predict(X)` - Get class predictions
  - `predictProba(X)` - Get class probabilities

- `Autoencoder(input_dim, encoding_dim)` - Unsupervised learning
  - `fit(X)` - Train reconstruction
  - `transform(X)` - Encode to latent space
  - `inverseTransform(Z)` - Decode from latent space

### Decomposition
- `PCA(n_components)` - Principal component analysis
  - `fit(X)` - Learn components
  - `transform(X)` - Project onto components
  - `fitTransform(X)` - Learn and project
  - `inverseTransform(X)` - Reconstruct
  - `getExplainedVarianceRatio()` - Variance explained

## Integration with Other Modules

The sklearn module integrates seamlessly with:
- **NumJa**: Array operations and linear algebra
- **Pandas**: DataFrames for data manipulation
- **Matplotlib/Seaborn**: Visualization

### Example: Full ML Pipeline
```java
// 1. Load data with Pandas
DataFrame df = Pandas.read_csv("data.csv");
NDArray X = ...;  // Extract features

// 2. Preprocess with sklearn
StandardScaler scaler = new StandardScaler();
X_scaled = scaler.fitTransform(X);

// 3. Reduce dimensions with PCA
PCA pca = new PCA(10);
X_reduced = pca.fitTransform(X_scaled);

// 4. Cluster or classify
KMeans kmeans = new KMeans(5);
kmeans.fit(X_reduced);

// 5. Visualize with Matplotlib
Matplotlib.scatter(X_reduced.get(0), X_reduced.get(1));
Matplotlib.show();
```

## Performance Notes

- **KMeans**: O(n*k*d*iter) - efficient for large datasets
- **GMM**: O(n*k*d*iter) - more computation than KMeans due to covariance
- **PCA**: O(n*d^2) - covariance computation
- **MLP**: Depends on network size and iterations
- **Autoencoder**: Depends on architecture and training iterations

## Future Enhancements

- [ ] SVM (Support Vector Machines)
- [ ] Random Forests
- [ ] Gradient Boosting
- [ ] RNN (Recurrent Neural Networks)
- [ ] Convolutional layers
- [ ] Batch normalization
- [ ] Dropout regularization
- [ ] Model serialization/persistence
