package com.numja.pandas.internal;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import pandas.DataFrame;
import pandas.Series;
import pandas.internal.ChunkedReadOptions;
import pandas.internal.CsvChunkReader;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.NoSuchElementException;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class CsvChunkReaderTest {

    private Path tempPath;

    @Before
    public void setUp() throws IOException {
        tempPath = Files.createTempFile("phase4_csv_", ".csv");
    }

    @After
    public void tearDown() throws IOException {
        Files.deleteIfExists(tempPath);
    }

    private void write(String content) throws IOException {
        Files.writeString(tempPath, content, StandardCharsets.UTF_8);
    }

    @Test
    public void chunkCountMatchesCeil() throws IOException {
        StringBuilder sb = new StringBuilder("a,b\n");
        for (int i = 0; i < 25; i++) sb.append(i).append(",").append(i * 2).append("\n");
        write(sb.toString());

        ChunkedReadOptions opts = new ChunkedReadOptions().chunkRows(10);
        try (CsvChunkReader reader = new CsvChunkReader(tempPath, opts)) {
            assertTrue(reader.hasNext());
            DataFrame c1 = reader.next();
            assertEquals(10, c1.shape()[0]);
            assertTrue(reader.hasNext());
            DataFrame c2 = reader.next();
            assertEquals(10, c2.shape()[0]);
            assertTrue(reader.hasNext());
            DataFrame c3 = reader.next();
            assertEquals(5, c3.shape()[0]);
            assertFalse(reader.hasNext());
        }
    }

    @Test
    public void close_releasesFileHandle() throws IOException {
        write("a,b\n1,2\n3,4\n");
        ChunkedReadOptions opts = new ChunkedReadOptions().chunkRows(10);
        CsvChunkReader reader = new CsvChunkReader(tempPath, opts);
        reader.close();
        // Re-open on Windows succeeds only if the previous handle was released.
        try (BufferedReader br = Files.newBufferedReader(tempPath, StandardCharsets.UTF_8)) {
            assertNotNull(br.readLine());
        }
    }

    @Test
    public void pathTraversal_canonicalized() throws IOException {
        write("a,b\n1,2\n3,4\n");
        Path normalized = Paths.get(tempPath.toString()).normalize();
        ChunkedReadOptions opts = new ChunkedReadOptions().chunkRows(10);
        try (CsvChunkReader reader = new CsvChunkReader(normalized, opts)) {
            assertTrue(reader.hasNext());
            DataFrame c = reader.next();
            assertEquals(2, c.shape()[0]);
        }
    }

    @Test
    public void emptyFile_throwsIAE() throws IOException {
        write("a,b\n");
        ChunkedReadOptions opts = new ChunkedReadOptions().chunkRows(10);
        try {
            new CsvChunkReader(tempPath, opts);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue("message should mention 'File CSV rong': " + e.getMessage(),
                e.getMessage().contains("File CSV rong"));
        }
    }

    @Test
    public void nanOnBadNumeric() throws IOException {
        write("a,b\n1,foo\n2,3.5\n");
        ChunkedReadOptions opts = new ChunkedReadOptions().chunkRows(10);
        try (CsvChunkReader reader = new CsvChunkReader(tempPath, opts)) {
            DataFrame c = reader.next();
            Series a = c.getColumn("a");
            Series b = c.getColumn("b");
            assertEquals(1.0, a.get(0), 1e-9);
            assertEquals(2.0, a.get(1), 1e-9);
            assertTrue("b.get(0) should be NaN", Double.isNaN(b.get(0)));
            assertEquals("foo", b.getString(0));
            assertEquals(3.5, b.get(1), 1e-9);
        }
    }

    @Test
    public void skiprowsAndIndexCol() throws IOException {
        write("skip line 1\nskip line 2\nname,age\nalice,30\nbob,25\ncarol,40\n");
        ChunkedReadOptions opts = new ChunkedReadOptions()
            .chunkRows(100)
            .skiprows(2)
            .index_col(0);
        try (CsvChunkReader reader = new CsvChunkReader(tempPath, opts)) {
            DataFrame c = reader.next();
            assertArrayEquals(new String[]{"age"}, c.columns());
            assertEquals("alice", c.index()[0]);
            assertEquals(30.0, c.getColumn("age").get(0), 1e-9);
            assertEquals(25.0, c.getColumn("age").get(1), 1e-9);
            assertEquals(40.0, c.getColumn("age").get(2), 1e-9);
        }
    }

    @Test
    public void exhaustedNext_throwsNoSuchElement() throws IOException {
        write("a,b\n1,2\n");
        ChunkedReadOptions opts = new ChunkedReadOptions().chunkRows(10);
        try (CsvChunkReader reader = new CsvChunkReader(tempPath, opts)) {
            reader.next();
            assertFalse(reader.hasNext());
            try {
                reader.next();
                fail("Expected NoSuchElementException");
            } catch (NoSuchElementException e) {
                // expected
            }
        }
    }
}
