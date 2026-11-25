import argparse

from ytpgen.cli import _source_ids


class FakeDB:
    def all_sources(self):
        return [{"id": 1, "video_id": "DuckandC1951"}, {"id": 2, "video_id": "local-decoder"}, {"id": 3, "video_id": "ShyGuy1947"}]


def test_clean_tier_and_films(tmp_path):
    (tmp_path / "clean_sources.txt").write_text("DuckandC1951  # Prelinger, PD\nShyGuy1947\n")
    cfg = {"workspace": str(tmp_path)}
    ns = lambda **k: argparse.Namespace(**{"films": None, "clean": False, **k})
    assert _source_ids(ns(), cfg, FakeDB()) is None
    assert _source_ids(ns(clean=True), cfg, FakeDB()) == {1, 3}
    assert _source_ids(ns(clean=True, films=["Shy", "local"]), cfg, FakeDB()) == {3}
    (tmp_path / "clean_sources.txt").unlink()
    assert _source_ids(ns(clean=True), cfg, FakeDB()) == set()
