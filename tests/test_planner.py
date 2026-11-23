"""v0.2 phrase inventory, planner resolution/cost, and WER (no LLM, no media)."""

from __future__ import annotations

import json

import pytest

from ytpgen.compose.matcher import Matcher
from ytpgen.compose.phrases import derive_phrases
from ytpgen.compose.planner import make_plans, phrase_clip, plan_cost
from ytpgen.compose.verify import wer
from ytpgen.index import Index
from ytpgen.models import Clip


@pytest.fixture()
def db(tmp_path):
    idx = Index(tmp_path / "lib.db")
    a = idx.add_source("a", "", "A", "/media/a.mp4", 60.0)
    b = idx.add_source("b", "", "B", "/media/b.mp4", 60.0)
    # a: "duck and cover" / pause / "the bomb is coming"
    for w, s, e in [("duck", 1.0, 1.3), ("and", 1.35, 1.5), ("cover", 1.55, 2.0),
                    ("the", 3.0, 3.1), ("bomb", 3.15, 3.5), ("is", 3.55, 3.7), ("coming", 3.75, 4.2)]:
        idx.add_word(a, w, s, e)
    idx.add_scene(a, 0.0, 10.0, "shot", 1.0)
    # b: "i love you" then a scene cut splits "very much"
    for w, s, e in [("i", 1.0, 1.1), ("love", 1.15, 1.4), ("you", 1.45, 1.7), ("very", 1.8, 2.0), ("much", 2.05, 2.4)]:
        idx.add_word(b, w, s, e)
    idx.add_scene(b, 0.0, 1.75, "shot", 1.0)
    idx.add_scene(b, 1.75, 10.0, "shot", 1.0)
    idx.commit()
    Matcher._ipa_infer = lambda token: None  # no espeak in tests
    yield idx
    Matcher._ipa_infer = None
    idx.close()


def test_phrases_split_on_pause_and_scene(db):
    texts = [p.text for p in derive_phrases(db.conn)]
    assert texts == ["duck and cover", "the bomb is coming", "i love you", "very much"]


def test_phrase_clip_pads_into_silence_only(db):
    ph = derive_phrases(db.conn)[0]  # duck and cover, next phrase starts 3.0
    full = phrase_clip(ph)
    assert full.start_s == pytest.approx(0.96) and full.end_s == pytest.approx(2.08)
    sub = phrase_clip(ph, 0, 1)  # "duck": tail clamped before "and" at 1.35
    assert sub.end_s == pytest.approx(1.33)


def test_plan_cost_prefers_long_phrases():
    one = [Clip(local_path="x", start_s=0, end_s=2, text="the bomb is coming", kind="phrase")]
    cut = [Clip(local_path="x", start_s=0, end_s=1, text=w, kind="word") for w in "the bomb is coming".split()]
    assert plan_cost(one, 0) < plan_cost(cut, 0) < plan_cost(cut, 1)


def test_make_plans_resolves_and_ranks(db):
    phrases = derive_phrases(db.conn)
    drafts = [
        {"title": "glue", "lines": [{"say": "bomb"}, {"say": "love"}, {"say": "zebra"}]},
        {"title": "whole", "lines": [{"p": 2}, {"p": 1}, {"p": 0, "words": "duck"}]},
    ]
    llm = lambda cfg, msgs, temperature, seed: json.dumps(drafts[seed])
    plans = make_plans({}, db, phrases, "deep", None, n=2, llm=llm)
    assert [p.title for p in plans] == ["whole", "glue"]
    best = plans[0]
    assert best.intended == "i love you the bomb is coming duck"
    assert [c.kind for c in best.clips] == ["phrase"] * 3
    assert plans[1].missing == 1  # "zebra" isn't in the library


def test_make_plans_found_only_drops_llm_words(db):
    phrases = derive_phrases(db.conn)
    seen = []
    draft = {"title": "mixed", "lines": [{"p": 2}, {"say": "love"}, {"p": 1}]}

    def llm(cfg, msgs, temperature, seed):
        seen.append(msgs[0]["content"])
        return json.dumps(draft)

    plans = make_plans({}, db, phrases, "deep", None, n=1, llm=llm, found_only=True)
    assert '"say"' not in seen[0] and "Never add words of your own" in seen[0]
    assert plans[0].intended == "i love you the bomb is coming"  # the glue "love" was dropped
    assert all(c.kind == "phrase" for c in plans[0].clips)


def test_wer():
    assert wer("the bomb is coming", "the bomb is coming") == 0
    assert wer("the bomb is coming", "the bomb coming") == pytest.approx(0.25)
    assert wer("a b", "") == 1.0


def test_plan_cost_penalises_retelling_the_film():
    a = Clip(local_path="x", start_s=0, end_s=2, text="one two", kind="phrase")
    nxt = Clip(local_path="x", start_s=2.5, end_s=4, text="three four", kind="phrase")
    far = Clip(local_path="x", start_s=60, end_s=62, text="three four", kind="phrase")
    assert plan_cost([a, far], 0) < plan_cost([a, nxt], 0)


def test_judge_unexpected_congruence(db):
    from ytpgen.compose.judge import judge

    by_id = {p.id: p for p in derive_phrases(db.conn)}
    seen = {}

    def llm(cfg, msgs, temperature, seed):
        seen["prompt"] = msgs[0]["content"]
        return json.dumps({"shift": [10, 5, 99], "fit": [8, 2], "why": "ok"})

    r = judge({}, [{"p": 2}, {"p": 1, "words": "bomb is coming"}, {"say": "duck"}], by_id, llm=llm)
    assert r["shift"] == [10, 5, 10] and r["fit"] == [8, 2]
    assert r["unexpected"] == pytest.approx(25 / 30, abs=1e-3) and r["congruent"] == pytest.approx(0.5)
    assert r["score"] == pytest.approx(round(25 / 30 * 0.5, 3))
    assert "[i love you] very much" not in seen["prompt"]  # scene cut: different context
    assert "bomb is coming" in seen["prompt"] and "no original scene" in seen["prompt"]


def test_scene_tags_flow_into_phrases_listing_and_judge(db):
    from ytpgen.analyze.tag import parse_tags, tag_scenes
    from ytpgen.compose.judge import judge
    from ytpgen.compose.planner import _listing

    replies = iter(['{"people": ["boy at desk"], "speaking": true, "setting": "classroom", "shot": "medium"}',
                    "not json", '{"people": ["woman in hat", "man"], "setting": "office", "shot": "close-up"}'])
    n = tag_scenes({}, db.conn, llm=lambda cfg, img: next(replies), grab=lambda path, at: b"jpg")
    assert n == 3
    assert parse_tags("garbage") == {"people": [], "speaking": False, "setting": "", "shot": ""}
    phrases = derive_phrases(db.conn)
    assert phrases[0].extra["visual"]["people"] == ["boy at desk"]
    listing = _listing(phrases)
    assert "## film 1, scene 0: classroom, medium: boy at desk, someone talking" in listing
    assert listing.count("## film 1, scene 0") == 1  # one header for both phrases of that scene

    by_id = {p.id: p for p in phrases}
    seen = {}

    def llm(cfg, msgs, temperature, seed):
        seen["p"] = msgs[0]["content"]
        return json.dumps({"shift": [10, 10], "fit": [10], "look": [0]})

    r = judge({}, [{"p": 0}, {"p": 3}], by_id, llm=llm)
    assert "on_screen" in seen["p"] and r["congruent"] == pytest.approx(0.5)


def test_plan_cost_penalises_one_film_dominating():
    mk = lambda sid, t: Clip(local_path=str(sid), start_s=t, end_s=t + 1, source_id=sid, text="a b", kind="phrase")
    mono = [mk(1, 0), mk(1, 100), mk(1, 200), mk(1, 300)]
    mixed = [mk(1, 0), mk(2, 0), mk(1, 300), mk(3, 0)]
    assert plan_cost(mixed, 0) < plan_cost(mono, 0)


def test_phrase_clip_never_zero_length(db):
    from ytpgen.compose.phrases import Phrase

    ph = Phrase(0, 1, "/m.mp4", ["but", "no"], [5.0, 5.0], [5.0, 5.2], prev_end_s=4.99, next_start_s=5.21)
    c = phrase_clip(ph, 0, 1)  # zero-length "but", hemmed in by neighbours
    assert c.end_s - c.start_s >= 0.15
