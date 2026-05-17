package numja.tests;

import org.junit.Test;

/**
 * JUnit wrapper to execute the legacy test runner in CI.
 */
public class NumJaJUnitTest {
    @Test
    public void runAll() {
        NumJaTest.main(new String[0]);
    }
}

