# ytpgen — YouTube Poop generator

Pipeline: bulk-acquire → segment → analyze (words + IPA/ARPABET phones) → match → compose (EDL) → render (pure ffmpeg).

Layout:
- src/ytpgen/acquire/ingest.py — playlist/channel expansion + yt-dlp bulk download
- src/ytpgen/analyze/segment.py — ffmpeg scene detection + faster-whisper word timestamps
- src/ytpgen/analyze/phonemes/ — IPA↔ARPABET mapping; benchmark engines (whisper_g2p, allosaurus, MFA)
- src/ytpgen/compose/matcher.py — text→clips (word/phone/syllable) and auto-poop planner
- src/ytpgen/compose/edl.py, renderer.py — EDL save/load + pure-ffmpeg render
- src/ytpgen/web/ — local Flask UI: browse library, preview clips, edit EDL, render
- scripts/bench_phonemes.py — engine benchmark harness (WER/phone-accuracy vs runtime)
- config/ytp.yaml — overrides src/ytpgen/config.py DEFAULTS
- Workspace for media/db/outputs: /Volumes/evol/ytpgen

Phoneme engine decision (benchmarked on 770s of real speech, /Volumes/evol/ytpgen/bench/):
- whisper_g2p: 3.25 min/min, dictionary-quality phones, word timing trustworthy, phone timing = even split (coarse)
- allosaurus uni2005: 0.92 min/min (3.5x faster), native phone timing, own phone inventory (naive ARPABET map scored 0.31 agreement — mapping artifact, not pure accuracy signal)
DECISION: whisper_g2p stays the index backbone; v2 adds allosaurus as a timing refiner (snap its boundaries to whisper word spans). MFA deferred (heavy Kaldi dep).

Known hardening note: Renderer._concat_reencode probes audio presence from segments[0] only — safe because every rendered segment is guaranteed an aac track (audio-less events fail in _render_event and are skipped), but revisit if a future producer can emit mixed-layout segments.

Seed code reference: ~/repos/video-desgin/src/ytp/ (word-level generator, moviepy-based) — concepts lifted, rebuilt on ffmpeg.