package numja.config;

import java.util.HashMap;
import java.util.Map;

/**
 * Manages thread pool configuration for NumJa operations.
 * Provides automatic and manual thread optimization with sensible defaults.
 */
public class ThreadPoolConfig {
    private static ThreadPoolConfig instance;
    
    private final int systemMaxThreads;
    private final int recommendedThreads;
    private int currentThreads;
    private boolean autoOptimize;
    
    private ThreadPoolConfig() {
        this.systemMaxThreads = Runtime.getRuntime().availableProcessors();
        this.recommendedThreads = calculateOptimalThreads();
        this.currentThreads = this.recommendedThreads;
        this.autoOptimize = true;
    }
    
    /**
     * Get singleton instance
     */
    public static synchronized ThreadPoolConfig getInstance() {
        if (instance == null) {
            instance = new ThreadPoolConfig();
        }
        return instance;
    }
    
    /**
     * Calculate optimal number of threads (60% of system max)
     */
    private int calculateOptimalThreads() {
        return Math.max(1, (int) (systemMaxThreads * 0.6));
    }
    
    /**
     * Get maximum allowed threads (60% of system max)
     */
    public int getMaxThreads() {
        return recommendedThreads;
    }
    
    /**
     * Get system total available threads
     */
    public int getSystemMaxThreads() {
        return systemMaxThreads;
    }
    
    /**
     * Get recommended number of threads
     */
    public int getRecommendedThreads() {
        return recommendedThreads;
    }
    
    /**
     * Get current thread count
     */
    public int getCurrentThreads() {
        return currentThreads;
    }

    /**
     * Python-style alias for getCurrentThreads.
     */
    public int get_threads() {
        return getCurrentThreads();
    }
    
    /**
     * Set the number of threads to use
     * @param numThreads Number of threads. If null or -1, use recommended.
     *                   Capped at 60% of system max.
     */
    public void setThreads(Integer numThreads) {
        if (numThreads == null || numThreads == -1) {
            this.currentThreads = recommendedThreads;
            this.autoOptimize = true;
        } else {
            int maxAllowed = (int) (systemMaxThreads * 0.6);
            this.currentThreads = Math.max(1, Math.min(numThreads, maxAllowed));
            this.autoOptimize = false;
        }
        updateEnvironment();
    }

    /**
     * Python-style alias for setThreads.
     */
    public void set_threads(Integer numThreads) {
        setThreads(numThreads);
    }
    
    /**
     * Enable automatic thread optimization
     */
    public void enableAutoOptimize() {
        this.autoOptimize = true;
        this.currentThreads = recommendedThreads;
        updateEnvironment();
    }
    
    /**
     * Disable automatic thread optimization
     */
    public void disableAutoOptimize() {
        this.autoOptimize = false;
    }
    
    /**
     * Check if auto optimization is enabled
     */
    public boolean isAutoOptimize() {
        return autoOptimize;
    }
    
    /**
     * Get configuration as map
     */
    public Map<String, Object> getConfigDict() {
        Map<String, Object> config = new HashMap<>();
        config.put("system_max_threads", systemMaxThreads);
        config.put("max_allowed_threads", recommendedThreads);
        config.put("recommended_threads", recommendedThreads);
        config.put("current_threads", currentThreads);
        config.put("auto_optimize", autoOptimize);
        return config;
    }

    /**
     * Python-style alias for getConfigDict.
     */
    public Map<String, Object> info() {
        return getConfigDict();
    }
    
    /**
     * Reset configuration to defaults
     */
    public void resetToDefaults() {
        this.currentThreads = recommendedThreads;
        this.autoOptimize = true;
        updateEnvironment();
    }
    
    /**
     * Update environment variables for LAPACK/BLAS
     */
    private void updateEnvironment() {
        String threadStr = String.valueOf(currentThreads);
        try {
            // These can't be set after JVM start, but we track for external tools
            System.setProperty("numja.threads", threadStr);
            System.setProperty("OPENBLAS_NUM_THREADS", threadStr);
            System.setProperty("MKL_NUM_THREADS", threadStr);
            System.setProperty("OMP_NUM_THREADS", threadStr);
        } catch (Exception e) {
            // Silently ignore if can't set properties
        }
    }
    
    @Override
    public String toString() {
        return String.format(
            "ThreadPoolConfig{system=%d, recommended=%d, current=%d, auto=%b}",
            systemMaxThreads, recommendedThreads, currentThreads, autoOptimize
        );
    }
}

