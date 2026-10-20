"""T2 인수 테스트 — 에이전트는 이 파일을 보지 못한다."""

from pathlib import Path

import yaml

from src.registry import check_class_counts

REPORT = {"per_class": {"train": {"daisy": 554, "dandelion": 785, "roses": 380, "sunflowers": 610, "tulips": 399},
                        "val": {"daisy": 10}, "test": {"daisy": 10}}}


def test_below_minimum_sorted():
    assert check_class_counts(REPORT, 400) == ["roses", "tulips"]


def test_none_below():
    assert check_class_counts(REPORT, 300) == []


def test_boundary_is_ok():
    assert check_class_counts(REPORT, 380) == []


def test_params():
    assert yaml.safe_load(open("params.yaml", encoding="utf-8"))["data_gate"]["min_train_per_class"] == 400


def test_step_calls_check():
    src = Path("pipeline/steps.py").read_text(encoding="utf-8")
    assert "check_class_counts" in src and "RuntimeError" in src
