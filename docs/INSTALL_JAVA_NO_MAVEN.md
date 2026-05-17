# Install & Use NumJa (Java-only, no Maven / no network)

This document explains how a client with only Java installed can use NumJa without Maven, Gradle, or internet access.

Options (choose one):

- **Fat JAR (recommended)** — a single JAR contains NumJa and all dependencies. Client only needs that one file.
- **Distributable ZIP (one-time install)** — ZIP contains JARs + installer script that copies jars into a per-user folder.

1) Fat JAR (single file)

Steps for the client:

1. Copy `numja-0.1.0.jar` to the client machine (file produced by the build and available at `target/numja-0.1.0.jar`).
2. Compile your app (if you have .java sources):

```powershell
javac -cp "C:\path\to\numja-0.1.0.jar;." -d out src\com\example\Main.java
```

3. Run your app:

```powershell
java -cp "out;C:\path\to\numja-0.1.0.jar" com.example.Main
```

If your app is already packaged as `your-app.jar`:

```powershell
java -cp "C:\path\to\your-app.jar;C:\path\to\numja-0.1.0.jar" com.example.Main
```

2) Distributable ZIP (one-time install; good for non-technical users)

As maintainer: produce `dist/numja-0.1.0.zip` (already included in this repo).

Client steps:

1. Extract `numja-0.1.0.zip` to a folder.
2. Run the included installer script (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File .\install_from_zip.ps1
```

This copies all JARs into `%USERPROFILE%\libs\numja\0.1.0`. After that the client can reference that folder.

3) Using the installed folder in projects

- Gradle (client `build.gradle`):

```groovy
repositories { flatDir { dirs System.getenv('USERPROFILE') + '/libs/numja/0.1.0' } }
dependencies { implementation name: 'numja-0.1.0' }
```

- Command-line run:

```powershell
java -cp "C:\path\to\your-app.jar;C:\Users\<user>\libs\numja\0.1.0\numja-0.1.0.jar" com.example.Main
```

- IntelliJ/Eclipse: add `%USERPROFILE%\libs\numja\0.1.0` as a project/global library.

Notes
- Both options avoid requiring the client to have Maven/Gradle or internet access.
- Prefer the Fat JAR for the simplest distribution (single file). Use the ZIP installer when you want a central per-user install for multiple projects on the same machine.

Questions or want me to prepare a runnable example `Main` and a ready-to-distribute ZIP in `dist/`? Reply with `example` or `dist`.
