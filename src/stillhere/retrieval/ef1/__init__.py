"""Evidence-First Retrieval v1 (EF1): the shared long-term-memory retrieval core.

    question + ask_time
      -> query.prepare          ask-time prefix separated from the body; ask time vs. event time kept apart
      -> temporal.resolve       event dates, ranges, offsets ("34天后", "后的第N天"), intervals; anchors that need a lookup
      -> shape.detect           question form from its words only (list / ordering / plan check / count / range summary ...)
      -> channels               raw messages (FTS5 bm25), memory statements (bm25), structured memory (B2 channels, capped),
                                time windows, dense vectors (optional) - every one read at known_at = as_of = ask time
      -> fusion.fuse            weighted reciprocal-rank fusion on the evidence unit (one source record / one dialogue)
      -> CandidateSet           ranked units with their messages, linked memories, channels, ranks and scores

Nothing here reads a benchmark field (question type, gold evidence, reference answer): the inputs are the question text, the persona,
the ask time and the store. The LifeBench adapter is one caller; the product runtime can call the same ``retrieve``.
"""
EF1_VERSION = "stillhere.retrieval.ef1/1.0"
# entry point: stillhere.retrieval.ef1.pipeline.retrieve / EF1Config (the package itself imports nothing, so no import cycle)
