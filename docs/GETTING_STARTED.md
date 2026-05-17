# 🚀 BẮT ĐẦU VỚI NUMJA (5 PHÚT)

## Bước 1: Tải ZIP (30 giây)

```
Tải: dist/numja-0.1.0.zip
```

## Bước 2: Giải Nén (30 giây)

```powershell
Expand-Archive -Path 'numja-0.1.0.zip' -DestinationPath 'C:\numja'
cd 'C:\numja'
```

## Bước 3: Chạy Script Cài (10 giây)

**Chọn 1 trong 2:**

### Cách A: Cài vào Folder (Dễ nhất)
```powershell
.\install_from_zip.ps1
```
✅ Xong! JAR được copy vào: `%USERPROFILE%\libs\numja\0.1.0\`

### Cách B: Cài vào Maven (Nếu dùng Maven)
```powershell
.\install_into_m2.ps1
```
✅ JAR được import vào: `~\.m2\repository\`

## Bước 4: Tạo File Java (2 phút)

**Tạo file `Test.java`:**
```java
import com.numja.NumJa;

public class Test {
    public static void main(String[] args) {
        var a = NumJa.array(1, 2, 3);
        var b = NumJa.array(4, 5, 6);
        var c = NumJa.add(a, b);
        System.out.println("a + b = " + c);
    }
}
```

## Bước 5: Chạy (1 phút)

```powershell
# Tạo classpath
$cp = (Get-ChildItem "$env:USERPROFILE\libs\numja\0.1.0\*.jar" | 
       ForEach-Object {$_.FullName}) -join ';'

# Compile
javac -cp "$cp;." Test.java

# Chạy
java -cp "$cp;." Test
```

## Kết quả:
```
a + b = NDArray(shape=[3])
[5.0]
[7.0]
[9.0]
```

✅ **XONG! Bạn đã chạy NumJa thành công!**

---

## Tiếp Theo?

- Xem thêm: [README.md](README.md)
- Ví dụ đầy đủ: `examples/NumJaClientExample.java`
- Hướng dẫn chi tiết: `QUICK_START_OFFLINE.md`

## API Nhanh

```java
// Tạo mảng
NumJa.array(1, 2, 3)
NumJa.linspace(0, 10, 100)
NumJa.zeros(3, 4)

// Toán học
NumJa.add(a, b)
NumJa.sin(x)
NumJa.sum(a)

// Ma trận
NumJa.det(A)
NumJa.inv(A)
NumJa.solve(A, b)

// Vẽ đồ thị
Matplotlib.plotLine(x, y, "title", "plot.png")
```

---

**Câu hỏi? Xem README.md**
