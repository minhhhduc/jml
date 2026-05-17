package sklearn.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * Parallel execution utilities for handling n_jobs.
 */
public class ParallelUtils {

    /**
     * Get the optimal number of threads based on n_jobs parameter.
     * @param n_jobs Number of threads. If 0 or omitted, use 60% of available processors.
     *               If -1, use all available processors.
     *               Otherwise, use exactly n_jobs.
     * @return Positive integer representing the number of threads to use.
     */
    public static int getNumThreads(int n_jobs) {
        int maxCores = Runtime.getRuntime().availableProcessors();
        if (n_jobs == -1) {
            return maxCores;
        } else if (n_jobs <= 0) {
            // Default 60% of available cores
            int defaultCores = (int) Math.round(maxCores * 0.6);
            return Math.max(1, defaultCores);
        } else {
            return Math.max(1, Math.min(n_jobs, maxCores));
        }
    }

    /**
     * Executes a list of tasks in parallel using a fixed thread pool.
     * @param tasks The tasks to execute.
     * @param n_jobs The n_jobs parameter.
     * @param <T> Return type of the tasks.
     * @return List of results in the order the tasks were submitted.
     */
    public static <T> List<T> execute(List<Callable<T>> tasks, int n_jobs) {
        int numThreads = getNumThreads(n_jobs);
        
        // Optimize: if only 1 thread or 1 task, run sequentially to avoid overhead
        if (numThreads == 1 || tasks.size() <= 1) {
            List<T> results = new ArrayList<>(tasks.size());
            for (Callable<T> task : tasks) {
                try {
                    results.add(task.call());
                } catch (Exception e) {
                    throw new RuntimeException("Error executing task: " + e.getMessage(), e);
                }
            }
            return results;
        }

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        try {
            List<Future<T>> futures = executor.invokeAll(tasks);
            List<T> results = new ArrayList<>(futures.size());
            for (Future<T> future : futures) {
                results.add(future.get());
            }
            return results;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Parallel execution interrupted", e);
        } catch (ExecutionException e) {
            throw new RuntimeException("Task execution failed: " + e.getCause().getMessage(), e);
        } finally {
            executor.shutdownNow();
        }
    }
}
