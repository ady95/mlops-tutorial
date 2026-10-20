"""T1 인수 테스트 — 에이전트는 이 파일을 보지 못한다."""

import yaml

from src.registry import judge

CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
GATE = {"min_test_f1": 0.85, "max_latency_ms": 20, "max_f1_drop": 0.0, "require_clean_git": True,
        "min_class_f1": 0.80}


def tags(**per_class):
    t = {"test_f1_macro": "0.9200", "test_latency_ms": "9.0", "git_dirty": "False"}
    t |= {f"test_f1_{c}": per_class.get(c, "0.9000") for c in CLASSES}
    return t


def test_params_has_min_class_f1():
    assert yaml.safe_load(open("params.yaml", encoding="utf-8"))["gate"]["min_class_f1"] == 0.80


def test_all_classes_above_passes():
    checks = judge(tags(), None, GATE)
    assert all(ok for _, ok in checks)
    assert sum(1 for text, _ in checks if any(c in text for c in CLASSES)) >= 5


def test_one_class_below_fails_with_name():
    failed = [text for text, ok in judge(tags(roses="0.7000"), None, GATE) if not ok]
    assert len(failed) == 1 and "roses" in failed[0]


def test_missing_class_tag_fails():
    t = tags()
    del t["test_f1_tulips"]
    failed = [text for text, ok in judge(t, None, GATE) if not ok]
    assert any("tulips" in text for text in failed)


def test_without_rule_no_class_checks():
    gate = {k: v for k, v in GATE.items() if k != "min_class_f1"}
    assert not any(any(c in text for c in CLASSES) for text, _ in judge(tags(), None, gate))
