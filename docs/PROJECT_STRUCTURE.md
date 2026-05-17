# NumJa Multi-Language Project

Complete implementation of NumJa (NumPy-like Array Library) in both Python and Java with:
- NumPy-compatible interface
- Automatic thread optimization (default 60% of system CPU threads)
- LAPACK/BLAS integration via SciPy (Python) and EJML (Java)
- Full linear algebra support
- Configurable thread management

## Project Structure

```
java_ml/
├── src/
│   ├── numja/                          # Python implementation
│   │   ├── __init__.py                 # Package initialization
│   │   ├── config.py                   # Thread configuration (Python)
│   │   ├── array.py                    # Core ndarray class
│   │   ├── operations.py               # Element-wise operations
│   │   ├── linalg.py                   # Linear algebra (LAPACK)
│   │   ├── utils.py                    # Utility functions
│   │   ├── tests.py                    # Python tests
│   │   ├── README.md                   # Python module docs
│   │   └── pom.xml                     # Maven for Python wrapper (optional)
│   │
│   ├── main/java/com/numja/            # Java implementation
│   │   ├── NumJa.java                  # Main facade
│   │   ├── config/
│   │   │   └── ThreadPoolConfig.java   # Thread configuration (Java)
│   │   ├── core/
│   │   │   ├── NDArray.java            # Core array class
│   │   │   └── ArrayOps.java           # Array operations
│   │   └── linalg/
│   │       ├── LinAlg.java             # Linear algebra (EJML)
│   │       ├── SVDResult.java
│   │       ├── QRResult.java
│   │       └── EigenResult.java
│   │
│   └── test/java/com/numja/tests/
│       └── NumJaTest.java              # Java test suite
│
├── pom.xml                             # Java Maven build
├── setup.py                            # Python setup
├── README.md                           # Main project documentation
├── JAVA_README.md                      # Java-specific documentation
└── [this file]                         # Project structure guide

```

## Implementation Comparison

### Python Version
- **Location**: `src/numja/`
- **Build**: `pip install -r requirements.txt`
- **Core**: NumPy arrays with SciPy LAPACK
- **Thread Management**: Automatic 60% detection + environment variable control
- **Performance**: Depends on NumPy/SciPy native implementations

### Java Version
- **Location**: `src/main/java/com/numja/`
- **Build**: Maven (pom.xml)
- **Core**: EJML (Efficient Java Matrix Library)
- **Thread Management**: Automatic 60% detection + ThreadPoolConfig
- **Performance**: Pure Java, no native dependencies

## Features in Both Versions

### Array Creation
- ✅ `array()` - Create from data
- ✅ `zeros()` - Create zero arrays
- ✅ `ones()` - Create ones arrays
- ✅ `eye()` - Identity matrix
- ✅ `arange()` - Range array
- ✅ `linspace()` - Linear space
- ✅ `full()` - Filled array

### Element-wise Operations
- ✅ Arithmetic: `add`, `subtract`, `multiply`, `divide`, `power`
- ✅ Math: `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`
- ✅ Other: `abs`, `negate`

### Reductions
- ✅ `sum`, `mean`, `std`, `var`, `min`, `max`
- ✅ `argmin`, `argmax`
- ✅ `prod` (Python only)

### Linear Algebra
- ✅ Matrix operations: `inv`, `det`, `trace`, `matrix_rank`
- ✅ Decompositions: `svd`, `qr`, `eig`, `cholesky`
- ✅ Solving: `solve`, `lstsq` (Python), `pinv`
- ✅ Norms: `norm`, `norm1`, `normInf`

### Thread Management
- ✅ Automatic detection (60% of system max)
- ✅ Manual override support
- ✅ Environment variable configuration
- ✅ Persistent configuration

## Quick Start

### Python

```bash
# Install dependencies
pip install numpy scipy

# Use the library
import sys
sys.path.insert(0, 'src')
import numja as nj

# Create arrays
a = nj.array([1, 2, 3])
b = nj.zeros((3, 4))

# Operations
c = nj.matmul(a, b)
result = nj.svd(b)
```

### Java

```bash
# Build with Maven
cd java_ml
mvn clean compile

# Use in code
import com.numja.NumJa;
import com.numja.core.NDArray;

NDArray a = NumJa.array(1, 2, 3);
NDArray b = NumJa.zeros(3, 4);

NDArray c = NumJa.dot(a, b);
```

## Thread Configuration

Both versions provide identical thread management:

### Python
```python
from numja import ThreadPoolConfig, set_num_threads

config = ThreadPoolConfig()
print(config.get_config_dict())  # View configuration

set_num_threads(4)              # Set to 4 threads
set_num_threads(None)           # Reset to recommended
```

### Java
```java
import com.numja.NumJa;
import com.numja.config.ThreadPoolConfig;

ThreadPoolConfig config = NumJa.getThreadConfig();
System.out.println(config.getConfigDict());  // View configuration

NumJa.setNumThreads(4);         // Set to 4 threads
NumJa.setNumThreads(null);      // Reset to recommended
```

### Default Behavior
- **System Detection**: Reads `Runtime.availableProcessors()` (Java) or `multiprocessing.cpu_count()` (Python)
- **Optimal Count**: 60% of system max threads
- **Example (8-core system)**:
  - System max: 8 threads
  - Optimal: 5 threads (60% of 8) → rounded to 4 or 5 depending on calculation
  - Recommended: 4-5 threads
  - User can override: 1-4 threads (capped at 60%)

## Building

### Python Version
```bash
# Install for development
pip install -e .

# Run tests
python src/numja/tests.py

# Create distribution
python setup.py sdist bdist_wheel
```

### Java Version
```bash
# Compile
mvn clean compile

# Run tests
mvn test

# Create JAR
mvn package

# JAR location: target/numja-0.1.0.jar
```

## Testing

### Python
```bash
cd src/numja
python tests.py
```

### Java
```bash
mvn test
# Or
mvn test -Dtest=NumJaTest
```

## Performance Tips

1. **Default 60%**: Optimal balance between performance and system responsiveness
2. **Large Operations**: Parallelization benefits increase with array size
3. **Tuning**: Test with different thread counts for your workload
4. **Memory**: Monitor memory usage when using maximum threads

## Dependencies

### Python
- numpy >= 1.20.0
- scipy >= 1.7.0

### Java
- EJML (Efficient Java Matrix Library) >= 0.43.1
- Apache Commons Math >= 3.6.1
- JUnit 4 (for testing)

## Version Info
- **Version**: 0.1.0
- **Python**: 3.7+
- **Java**: 11+
- **Last Updated**: 2026-05-16

## Features Implemented

### ✅ Complete
- [x] Core array class (ndarray for Python, NDArray for Java)
- [x] Array creation functions
- [x] Element-wise operations
- [x] Reduction operations
- [x] Linear algebra (inv, det, rank, trace, solve)
- [x] Matrix decompositions (SVD, QR, Eigenvalue, Cholesky)
- [x] Automatic thread optimization
- [x] Thread configuration management
- [x] Comprehensive test suites
- [x] Full documentation

### 🔄 Future Enhancements
- [ ] GPU acceleration
- [ ] Distributed computing (Spark integration)
- [ ] Additional statistical functions
- [ ] Image processing utilities
- [ ] Integration with existing ML frameworks
- [ ] Performance benchmarking

## Troubleshooting

### Python Issues
- **Import error**: Ensure `src/` is in PYTHONPATH
- **Missing dependencies**: Run `pip install numpy scipy`
- **Performance**: Check thread count with `get_num_threads()`

### Java Issues
- **Maven not found**: Install Maven or use `./mvnw` (wrapper)
- **Compilation errors**: Ensure Java 11+ is installed
- **Memory errors**: Increase heap with `-Xmx4g`

## License

MIT License - Free for academic and commercial use

## Contributing

Contributions welcome! Submit issues, feature requests, or pull requests.

---

**This is a complete, production-ready implementation combining:**
- NumPy-like interface for familiarity
- Automatic thread optimization for performance
- LAPACK/BLAS support for mathematical rigor
- Multi-language support (Python + Java)
