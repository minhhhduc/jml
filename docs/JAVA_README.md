# NumJa Java Library

NumJa is a Java library providing NumPy-like array operations with automatic thread optimization and LAPACK-equivalent linear algebra operations using EJML.

## Features

- **NumPy-like Interface**: Familiar API that mimics NumPy
- **Automatic Thread Optimization**: Intelligent thread management for optimal performance
- **EJML Integration**: High-performance linear algebra via EJML (BLAS/LAPACK equivalent)
- **Thread Configuration**: Easy control over thread counts with sensible defaults (60% of system threads)
- **Pure Java**: No native dependencies needed

## Project Structure

```
java_ml/
├── src/
│   ├── main/java/com/numja/
│   │   ├── NumJa.java                    # Main facade
│   │   ├── config/
│   │   │   └── ThreadPoolConfig.java     # Thread management
│   │   ├── core/
│   │   │   ├── NDArray.java              # Core array class
│   │   │   └── ArrayOps.java             # Array operations
│   │   └── linalg/
│   │       ├── LinAlg.java               # Linear algebra
│   │       ├── SVDResult.java
│   │       ├── QRResult.java
│   │       └── EigenResult.java
│   └── test/java/com/numja/tests/
│       └── NumJaTest.java                # Test suite
├── pom.xml                               # Maven configuration
└── README.md                             # Documentation
```

## Quick Start

### Setup with Maven

```bash
# Compile
mvn clean compile

# Run tests
mvn test

# Package
mvn package
```

### Basic Usage

```java
import com.numja.NumJa;
import com.numja.core.NDArray;

public class Main {
    public static void main(String[] args) {
        // Create arrays
        NDArray a = NumJa.array(1, 2, 3, 4, 5);
        NDArray b = NumJa.zeros(3, 4);
        
        // Operations
        NDArray c = a.add(10);
        double result = NumJa.sum(a);
        
        System.out.println("Sum: " + result);
    }
}
```

## Array Creation

```java
// From values
NDArray a = NumJa.array(1, 2, 3, 4, 5);
NDArray b = NumJa.array(new double[][]{{1, 2}, {3, 4}});

// Special arrays
NDArray zeros = NumJa.zeros(3, 4);
NDArray ones = NumJa.ones(2, 3);
NDArray identity = NumJa.eye(4);

// Ranges
NDArray range = NumJa.arange(0, 10, 2);           // [0, 2, 4, 6, 8]
NDArray linspace = NumJa.linspace(0, 1, 50);     // 50 values from 0 to 1

// Full array
NDArray full = NumJa.full(3, 3, 5.0);             // 3x3 matrix filled with 5.0
```

## Element-wise Operations

```java
NDArray a = NumJa.array(1, 2, 3);
NDArray b = NumJa.array(4, 5, 6);

// Arithmetic
NDArray sum = NumJa.add(a, b);
NDArray diff = NumJa.subtract(a, b);
NDArray prod = NumJa.multiply(a, b);
NDArray div = NumJa.divide(a, b);
NDArray pow = NumJa.power(a, 2);

// Math functions
NDArray sqrt_a = NumJa.sqrt(a);
NDArray exp_a = NumJa.exp(a);
NDArray log_a = NumJa.log(a);
NDArray sin_a = NumJa.sin(a);
NDArray cos_a = NumJa.cos(a);
```

## Reductions

```java
NDArray data = NumJa.arange(1, 11, 1);

double sum = NumJa.sum(data);
double mean = NumJa.mean(data);
double std = NumJa.std(data);
double var = NumJa.var(data);
double min = NumJa.min(data);
double max = NumJa.max(data);

int minIdx = NumJa.argmin(data);
int maxIdx = NumJa.argmax(data);
```

## Linear Algebra

```java
NDArray A = NumJa.array(new double[][]{{1, 2}, {3, 4}});
NDArray B = NumJa.array(new double[][]{{5, 6}, {7, 8}});

// Matrix multiplication
NDArray C = NumJa.dot(A, B);

// Determinant and trace
double det = NumJa.det(A);
double trace = NumJa.trace(A);

// Inverse and rank
NDArray inv = NumJa.inv(A);
int rank = NumJa.matrixRank(A);

// Norms
double normF = NumJa.norm(A);        // Frobenius norm
double norm1 = NumJa.norm1(A);       // 1-norm
double normInf = NumJa.normInf(A);   // Infinity norm
```

## Matrix Decompositions

```java
NDArray A = NumJa.array(new double[][]{{1, 2}, {3, 4}});

// Singular Value Decomposition
SVDResult svd = NumJa.svd(A);
NDArray U = svd.getU();
double[] s = svd.getSingularValues();
NDArray Vt = svd.getV();

// QR Decomposition
QRResult qr = NumJa.qr(A);
NDArray Q = qr.getQ();
NDArray R = qr.getR();

// Eigenvalue Decomposition
EigenResult eigen = NumJa.eig(A);
double[] eigenvalues = eigen.getEigenvalues();
NDArray eigenvectors = eigen.getEigenvectors();

// Cholesky Decomposition (for positive definite matrices)
NDArray L = NumJa.cholesky(A);
```

## Solving Linear Systems

```java
// Solve Ax = b
NDArray A = NumJa.array(new double[][]{{3, 1}, {1, 2}});
NDArray b = NumJa.array(9, 8);

NDArray x = NumJa.solve(A, b);

// Pseudo-inverse
NDArray pinv = NumJa.pinv(A);
```

## Thread Optimization

NumJa automatically manages threading with sensible defaults.

```java
import com.numja.NumJa;
import com.numja.config.ThreadPoolConfig;

// Get configuration
ThreadPoolConfig config = NumJa.getThreadConfig();

// View current configuration
System.out.println(config.getConfigDict());
// Output: {
//     system_max_threads: 8,
//     max_allowed_threads: 4,        // 60% of 8
//     recommended_threads: 4,
//     current_threads: 4,
//     auto_optimize: true
// }

// Get current threads
int current = NumJa.getNumThreads();

// Set custom thread count (capped at 60% of system max)
NumJa.setNumThreads(2);

// Reset to recommended
NumJa.setNumThreads(null);

// Enable/disable auto-optimization
config.enableAutoOptimize();
config.disableAutoOptimize();

// Reset to defaults
config.resetToDefaults();
```

## Thread Configuration Details

### Default Behavior
- **System Analysis**: On first use, detects available CPU threads
- **Optimal Threads**: Default is 60% of available CPU threads
- **Automatic**: By default, thread optimization happens transparently
- **Overridable**: You can manually set thread count if needed

### Example on 8-core system:
- System max threads: 8
- Default recommended: 4 (60% of 8)  
- Maximum allowed: 4
- Can be customized: 1-4 threads

### Why 60%?
- Balances performance with system responsiveness
- Leaves room for other processes
- Optimal for most common workloads
- Prevents oversubscription

## Performance Considerations

1. **Parallelization**: EJML automatically parallelizes large matrix operations
2. **Thread Tuning**: Default 60% is optimal for most workloads
3. **Large Arrays**: Better performance with larger arrays due to parallelization
4. **Custom Tuning**: Test different thread counts for your specific workload

## Dependencies

- **EJML** (0.43.1): Efficient Java Matrix Library for linear algebra
- **Apache Commons Math** (3.6.1): Mathematical utilities
- **JUnit** (4.13.2): For testing

## Examples

### Matrix Computation

```java
import com.numja.NumJa;

// Create matrices
NDArray A = NumJa.arange(1, 13, 1).reshape(3, 4);
NDArray B = NumJa.arange(1, 13, 1).reshape(4, 3);

// Multiply
NDArray C = NumJa.dot(A, B);

// Factorize
SVDResult svd = NumJa.svd(A);
System.out.println("Singular values: " + java.util.Arrays.toString(svd.getSingularValues()));
```

### Statistical Analysis

```java
// Generate data
NDArray data = NumJa.arange(1, 101, 1);

// Statistics
double mean = NumJa.mean(data);
double std = NumJa.std(data);
double min = NumJa.min(data);
double max = NumJa.max(data);

System.out.println("Mean: " + mean + ", Std: " + std);
System.out.println("Range: [" + min + ", " + max + "]");
```

### Solving Linear System

```java
// Define system Ax = b
NDArray A = NumJa.array(new double[][]{{3, 1}, {1, 2}});
NDArray b = NumJa.array(9, 8);

// Solve
NDArray x = NumJa.solve(A, b);
System.out.println("Solution: " + x);
```

## Building from Source

```bash
# Clone/download the project
cd java_ml

# Build with Maven
mvn clean compile

# Run tests
mvn test

# Create JAR
mvn package

# JAR will be in target/
```

## Testing

```bash
# Run test suite
mvn test

# Or run specific test
mvn test -Dtest=NumJaTest
```

## Troubleshooting

### Out of Memory
- Reduce thread count: `NumJa.setNumThreads(2)`
- Use smaller matrix sizes
- Increase JVM heap: `java -Xmx4g YourApp`

### Performance Issues
- Check current thread count: `NumJa.getNumThreads()`
- Try different thread counts
- Ensure matrices are reasonably sized

### Decomposition Errors
- Check matrix properties (singular, non-square, etc.)
- For Cholesky: matrix must be positive definite
- For QR/SVD: any rectangular matrix works

## License

MIT License

## Contributing

Contributions are welcome! Areas for enhancement:
- More statistical functions
- Distributed computing support
- GPU acceleration
- Image processing utilities
- Integration with existing Java ML frameworks

---

**Version**: 0.1.0  
**Last Updated**: 2026-05-16
