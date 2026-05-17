-dontwarn
-dontnote

# Giữ lại các thuộc tính quan trọng để khi debug hoặc reflection không bị lỗi hoàn toàn
-keepattributes Exceptions,InnerClasses,Signature,Deprecated,SourceFile,LineNumberTable,*Annotation*,EnclosingMethod

# Keep Pandas module (everything public)
-keep public class pandas.** { public *; }

# Keep Plot modules
-keep public class matplotlib.** { public *; }
-keep public class seaborn.** { public *; }

# Keep Sklearn module for reflection
-keep public class sklearn.** { *; }

# Keep Core and LinAlg API
-keep public class numja.NumJa { public *; }
-keep public class numja.core.NDArray { public *; }
-keep public class numja.linalg.SVD { public *; }
-keep public class numja.linalg.QR { public *; }
-keep public class numja.linalg.Eigen { public *; }
-keep public class numja.linalg.Lstsq { public *; }
-keep public class numja.config.ThreadPoolConfig { public *; }

# Tùy chọn: Xóa các thông tin debug dư thừa (nếu muốn làm file nhẹ hơn)
#-assumenosideeffects class android.util.Log {
#    public static boolean isLoggable(java.lang.String, int);
#    public static int v(...);
#    public static int i(...);
#    public static int w(...);
#    public static int d(...);
#    public static int e(...);
#}
