from ytpgen.index import Index


def test_silent_source_counts_as_analyzed(tmp_path):
    """A source with scenes but no words (a silent film) must not be re-analyzed by `analyze --all`: that renumbers
    its scenes and breaks every scene id that refers to them."""
    idx = Index(tmp_path / "lib.db")
    silent = idx.add_source("golem", "", "The Golem", "/x.mp4")
    idx.add_source("fresh", "", "Not yet analyzed", "/y.mp4")
    idx.add_scene(silent, 0.0, 4.0, "scene", 1.0)
    assert idx.analyze_source_ids() == {silent}
