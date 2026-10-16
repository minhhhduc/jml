package pandas.internal;

/**
 * Configuration POJO for chunked CSV reads. Mirrors the shape of
 * {@code Pandas.ReadCsvOptions} (sep, skiprows, index_col) and adds
 * {@code chunkRows} which controls the size of each emitted
 * {@link pandas.DataFrame} chunk.
 *
 * <p>Validate via {@link #validate()} at the use site (not the ctor) so the
 * caller can attach path context to the error.
 *
 * <p>Trust boundary: caller-controlled options → in-process reader.
 * STRIDE threats: T405 OOM via pathologically large {@code chunkRows}
 * (mitigated by the {@code <= 100_000} ceiling in {@code validate()}).
 *
 * @since ASVS-L1
 */
public class ChunkedReadOptions {

    /** Hard ceiling to keep a single chunk from OOMing the JVM. */
    public static final int MAX_CHUNK_ROWS = 100_000;

    public String sep = ",";
    public int skiprows = 0;
    public Integer index_col = null;

    // ponytail: 0 default is intentionally invalid; validate() at use site rejects it.
    // bump to 1_000_000 once StreamingGroupAggregator per-chunk allocation is profiled.
    public int chunkRows = 0;

    public ChunkedReadOptions sep(String sep) { this.sep = sep; return this; }
    public ChunkedReadOptions skiprows(int skip) { this.skiprows = skip; return this; }
    public ChunkedReadOptions index_col(Integer index_col) { this.index_col = index_col; return this; }
    public ChunkedReadOptions chunkRows(int n) { this.chunkRows = n; return this; }

    /**
     * Validate option values. Throws {@link IllegalArgumentException} on bad input.
     */
    public void validate() {
        if (chunkRows < 1) {
            throw new IllegalArgumentException("chunkRows must be >= 1, got " + chunkRows);
        }
        if (chunkRows > MAX_CHUNK_ROWS) {
            throw new IllegalArgumentException("chunkRows must be <= " + MAX_CHUNK_ROWS + ", got " + chunkRows);
        }
        if (index_col != null && index_col < 0) {
            throw new IllegalArgumentException("index_col must be >= 0, got " + index_col);
        }
    }
}
