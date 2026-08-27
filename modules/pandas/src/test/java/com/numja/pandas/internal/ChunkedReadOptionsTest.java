package com.numja.pandas.internal;

import org.junit.Test;
import pandas.internal.ChunkedReadOptions;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class ChunkedReadOptionsTest {

    @Test
    public void chunkRows_defaultIsZero() {
        ChunkedReadOptions opts = new ChunkedReadOptions();
        assertEquals(0, opts.chunkRows);
        try {
            opts.validate();
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue("message should mention chunkRows >= 1: " + e.getMessage(),
                e.getMessage().contains("chunkRows must be >= 1"));
        }
    }

    @Test
    public void chunkRowsTooSmall_throwsIAE() {
        ChunkedReadOptions opts = new ChunkedReadOptions().chunkRows(0);
        try {
            opts.validate();
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("chunkRows must be >= 1"));
        }
    }

    @Test
    public void chunkRowsTooLarge_throwsIAE() {
        ChunkedReadOptions opts = new ChunkedReadOptions().chunkRows(200_000);
        try {
            opts.validate();
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("chunkRows must be <= 100000"));
        }
    }

    @Test
    public void negativeIndexCol_throwsIAE() {
        ChunkedReadOptions opts = new ChunkedReadOptions().chunkRows(10).index_col(-1);
        try {
            opts.validate();
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("index_col must be >= 0"));
        }
    }

    @Test
    public void validOptions_doesNotThrow() {
        ChunkedReadOptions opts = new ChunkedReadOptions().chunkRows(10_000).sep(";");
        opts.validate();
    }

    @Test
    public void fluentSetterReturnsThis() {
        ChunkedReadOptions opts = new ChunkedReadOptions();
        assertSame(opts, opts.chunkRows(100));
        assertSame(opts, opts.sep("|"));
        assertSame(opts, opts.skiprows(5));
        assertSame(opts, opts.index_col(2));
    }
}
