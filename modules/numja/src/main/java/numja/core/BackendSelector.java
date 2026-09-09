package numja.core;

import java.util.Set;

/**
 * Single source of truth for the active {@link ComputeBackend}.
 *
 * <p>Defaults to {@link CpuThreadBackend#getInstance()}. {@link #setBackend(String)}
 * honors an explicit allowlist + opt-in flag (HW-01 threat model). Unknown names
 * — even with the opt-in flag set — fall back silently to {@code CpuThreadBackend}
 * and emit a warning to {@code System.err}. This class never reflectively loads an
 * arbitrary class name: there is no class-loader dispatch, only a closed
 * {@code Set<String>} membership check.
 *
 * <p>The {@link #THRESHOLD_GPU} constant is declared for 05-02 (where dispatch
 * branches on size). It is intentionally unused in 05-01: the GPU path is never
 * auto-promoted in this phase.
 */
public final class BackendSelector {

    /**
     * Cell-count ceiling above which a GPU backend would be considered in 05-02.
     * Declared here per research; not consulted in 05-01.
     */
    public static final int THRESHOLD_GPU = 4_096_000;

    private static final Set<String> ALLOWLIST = Set.of("cpu-thread");

    private static volatile ComputeBackend active = CpuThreadBackend.getInstance();

    private BackendSelector() {}

    /** Returns the currently active backend. Never {@code null}. */
    public static ComputeBackend get() {
        return active;
    }

    /**
     * Attempts to set the active backend to the named one. Honors an explicit
     * allowlist and the {@code numja.backend.allow} opt-in flag. Unknown names
     * — or a missing opt-in flag — fall back to {@link CpuThreadBackend} and
     * log a warning. Never throws on an unknown name; never reflectively loads
     * a class.
     */
    public static void setBackend(String name) {
        ComputeBackend fallback = CpuThreadBackend.getInstance();
        boolean allowed = "true".equals(System.getProperty("numja.backend.allow"));
        if (allowed && name != null && ALLOWLIST.contains(name)) {
            if ("cpu-thread".equals(name)) {
                active = fallback;
                return;
            }
        }
        // Fallback path: opt-in missing, name unknown, or name null.
        if (name != null && !ALLOWLIST.contains(name)) {
            System.err.println("[BackendSelector] unknown backend name '" + name
                + "'; falling back to cpu-thread");
        }
        active = fallback;
    }
}
