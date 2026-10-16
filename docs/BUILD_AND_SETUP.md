# Build and Setup Instructions

> NOTE: This repo is Java-only. There is no Python package — the earlier
> installer/test-runner instructions for a `src/numja` Python tree were
> a leftover from a previous iteration and have been removed.
> Build and test with Maven on Windows PowerShell.

## Windows Setup (PowerShell + Maven)

### Prerequisites
- Java 17+ JDK (release jars compile with `--release 11`; Maven modules
  use source/target 17; running on newer JDKs works but the upper bound
  is unverified — see README)
- Maven 3.9.15 at `C:\Users\Admin\.maven\maven-3.9.15\bin`
  (not on default PATH — prefix it in every new shell)

### Installation
```powershell
# Navigate to project
cd D:\unknown\projects\java_ml

# Put Maven on PATH for this shell (required every new shell)
$env:Path = "C:\Users\Admin\.maven\maven-3.9.15\bin;" + $env:Path

# Verify
mvn -version
java -version
```

### Running Tests
```powershell
# Fast smoke: core modules only
mvn -pl modules/numja,modules/pandas,modules/sklearn test

# Include upstream deps explicitly
mvn -pl modules/sklearn -am test

# Single test class (PowerShell quoting)
mvn -pl modules/numja -am test "-Dtest=CpuThreadBackendGoldenTest"
```

### Building Jars
```powershell
mvn -pl modules/numja,modules/pandas,modules/sklearn -am package -DskipTests
```

## Java Setup

### Prerequisites
- Java 17+ JDK
- Maven 3.9.15 (`C:\Users\Admin\.maven\maven-3.9.15\bin`, prefixed to PATH)

### Build with Maven (PowerShell)

```powershell
# Navigate to project
cd D:\unknown\projects\java_ml
$env:Path = "C:\Users\Admin\.maven\maven-3.9.15\bin;" + $env:Path

# Clean and compile
mvn clean compile

# Run tests (module-scoped is faster)
mvn -pl modules/numja,modules/pandas,modules/sklearn test

# Create JARs
mvn package
```

> There is no Maven wrapper (`mvnw`) in this repo; always use the
> Maven 3.9.15 install above.

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

### Java Issues

**Maven not found (`mvn` is not recognized)**
```powershell
# Maven is NOT on default PATH on this machine; prefix it first:
$env:Path = "C:\Users\Admin\.maven\maven-3.9.15\bin;" + $env:Path
mvn -version
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
    runs-on: windows-latest
    strategy:
      matrix:
        java-version: [17, 25]
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-java@v2
        with:
          java-version: ${{ matrix.java-version }}
      - run: mvn -pl modules/numja,modules/pandas,modules/sklearn test
```

## Documentation

### Generate Documentation

**Java**
```bash
mvn javadoc:javadoc
# Output: target/site/apidocs/
```

## Version Management

### Java
- Edit version in `pom.xml` (`<version>` tag)
- Build and tag: `git tag v0.1.0`

## Performance Benchmarks

### Running Benchmarks

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
