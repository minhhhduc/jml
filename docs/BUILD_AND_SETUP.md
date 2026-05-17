# Build and Setup Instructions

## Python Setup

### Installation
```bash
# Navigate to project
cd java_ml

# Install dependencies
pip install numpy scipy

# Install package (optional)
pip install -e .
```

### Running Python Tests
```bash
# Direct execution
python src/numja/tests.py

# Or with imports
cd src/numja
python tests.py
```

### Using Python NumJa
```python
import sys
sys.path.insert(0, 'src')

import numja as nj
from numja import ThreadPoolConfig, set_num_threads

# Create arrays
a = nj.array([1, 2, 3, 4, 5])
b = nj.zeros((3, 4))

# Operations
c = a + 10
result = nj.matmul(a, b.T)

# Thread management
set_num_threads(4)
print(nj.get_num_threads())
```

## Java Setup

### Prerequisites
- Java 11 or higher
- Maven 3.6+ or use the bundled wrapper

### Build with Maven

```bash
# Navigate to Java project
cd java_ml

# Clean and compile
mvn clean compile

# Run tests
mvn test

# Create JAR
mvn package

# Run JAR
java -jar target/numja-0.1.0.jar
```

### Using Maven Wrapper (if available)
```bash
./mvnw clean compile
./mvnw test
./mvnw package
```

### Using in Java Project

Add to your `pom.xml`:
```xml
<dependency>
    <groupId>com.numja</groupId>
    <artifactId>numja</artifactId>
    <version>0.1.0</version>
</dependency>
```

Or copy the JAR to your classpath:
```bash
cp target/numja-0.1.0.jar /path/to/your/project/lib/
```

### Java Usage Example
```java
import com.numja.NumJa;
import com.numja.core.NDArray;
import com.numja.linalg.SVDResult;

public class Example {
    public static void main(String[] args) {
        // Create arrays
        NDArray a = NumJa.array(1, 2, 3, 4, 5);
        NDArray b = NumJa.zeros(3, 4);
        
        // Operations
        NDArray c = a.add(10);
        double sum = NumJa.sum(a);
        
        // Linear algebra
        NDArray A = NumJa.array(new double[][]{{1, 2}, {3, 4}});
        SVDResult svd = NumJa.svd(A);
        
        // Thread management
        NumJa.setNumThreads(4);
        System.out.println("Current threads: " + NumJa.getNumThreads());
    }
}
```

## Development Setup

### Python Development
```bash
# Install development dependencies
pip install pytest pytest-cov black flake8

# Format code
black src/numja/

# Lint code
flake8 src/numja/

# Run tests with coverage
pytest src/numja/tests.py --cov=src/numja
```

### Java Development
```bash
# Install IDE plugins
# - For IntelliJ IDEA: Built-in Maven support
# - For Eclipse: m2e plugin
# - For VS Code: Extension Pack for Java

# Generate IDE files (IntelliJ)
mvn idea:idea

# Generate IDE files (Eclipse)
mvn eclipse:eclipse
```

## Troubleshooting

### Python Issues

**ImportError: No module named 'numpy'**
```bash
pip install numpy scipy
```

**Module not found when importing numja**
```python
import sys
sys.path.insert(0, '/path/to/java_ml/src')
import numja
```

**Thread configuration not working**
- Restart Python interpreter after changing settings
- Check with `from numja import get_num_threads; print(get_num_threads())`

### Java Issues

**Maven not found**
```bash
# Install Maven or download from https://maven.apache.org
# Add to PATH, or use wrapper
./mvnw --version
```

**Compilation error: Java version incompatibility**
```bash
# Check Java version
java -version

# Update pom.xml if needed:
# <maven.compiler.source>11</maven.compiler.source>
# <maven.compiler.target>11</maven.compiler.target>
```

**Tests fail with "Out of Memory"**
```bash
# Increase JVM heap
mvn -DargLine="-Xmx2g" test
```

**JAR file too large**
- EJML and Commons Math are required dependencies
- Minimize JAR: `mvn package -DskipTests -P shade`

## Performance Tuning

### Python
```python
from numja import ThreadPoolConfig, set_num_threads

# View current setup
config = ThreadPoolConfig()
print(config.get_config_dict())

# Experiment with thread counts
for threads in [1, 2, 4, 8]:
    set_num_threads(threads)
    # Run your code and measure performance
```

### Java
```java
import com.numja.NumJa;
import com.numja.config.ThreadPoolConfig;

ThreadPoolConfig config = NumJa.getThreadConfig();
System.out.println(config.getConfigDict());

// Test different thread counts
for (int threads : new int[]{1, 2, 4, 8}) {
    NumJa.setNumThreads(threads);
    // Run your code and measure performance
}
```

## Deployment

### Python Package
```bash
# Create distribution
python setup.py sdist bdist_wheel

# Upload to PyPI (if configured)
twine upload dist/*
```

### Java Library
```bash
# Build JAR
mvn clean package

# Create source JAR
mvn source:jar-no-fork

# Create documentation
mvn javadoc:jar

# Maven Central deployment (if registered)
mvn deploy
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Build

on: [push, pull_request]

jobs:
  java:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        java-version: [11, 15, 17]
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-java@v2
        with:
          java-version: ${{ matrix.java-version }}
      - run: mvn clean test
  
  python:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -r requirements.txt
      - run: python -m pytest src/numja/tests.py
```

## Documentation

### Generate Documentation

**Java**
```bash
mvn javadoc:javadoc
# Output: target/site/apidocs/
```

**Python**
```bash
pip install sphinx
sphinx-build -b html docs/ docs/_build/
```

## Version Management

### Python
- Edit version in `setup.py`
- Tag release: `git tag v0.1.0`

### Java
- Edit version in `pom.xml` (`<version>` tag)
- Build and tag: `git tag v0.1.0`

## Performance Benchmarks

### Running Benchmarks

**Python**
```python
import time
import numja as nj

# Benchmark matrix multiplication
A = nj.arange(0, 10000).reshape(100, 100)
B = nj.arange(0, 10000).reshape(100, 100)

start = time.time()
for _ in range(100):
    C = nj.matmul(A, B)
end = time.time()

print(f"Time: {(end - start) / 100:.4f}s per operation")
```

**Java**
```java
import com.numja.NumJa;
import com.numja.core.NDArray;

NDArray A = NumJa.arange(0, 10000, 1).reshape(100, 100);
NDArray B = NumJa.arange(0, 10000, 1).reshape(100, 100);

long start = System.nanoTime();
for (int i = 0; i < 100; i++) {
    NumJa.dot(A, B);
}
long end = System.nanoTime();

System.out.println(String.format("Time: %.4f ms per operation", 
    (end - start) / 100.0 / 1_000_000));
```

---

For more detailed information, see:
- [README.md](README.md) - Main documentation
- [JAVA_README.md](JAVA_README.md) - Java-specific docs
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Project structure
