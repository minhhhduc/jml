package sklearn;

import org.junit.Test;

/**
 * JUnit wrapper to execute the legacy sklearn test runner in CI.
 */
public class TestSklearnJUnit {
    @Test
    public void runAll() {
        TestSklearn.main(new String[0]);
    }
}
