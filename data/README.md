# Generated index

`rag_index/` is a cache built from the configured document directory. It contains SQLite FTS5 chunks, a manifest, and optional NumPy vectors. The cache is ignored by version control and can be rebuilt with `python -m app.ingest --force`.
