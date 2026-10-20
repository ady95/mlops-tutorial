"""학습 코드 단위 테스트 (07-4). 사전학습 가중치를 내려받지 않고 구조만 확인한다."""

import torch

from src.training import build_model, data_version


def test_model_outputs_five_classes():
    for name in ("mobilenet_v3_small", "resnet18", "efficientnet_b0"):
        model = build_model(name, num_classes=5, freeze_backbone=False, pretrained=False).eval()
        with torch.no_grad():
            assert model(torch.randn(2, 3, 224, 224)).shape == (2, 5)


def test_freeze_backbone_trains_only_head():
    model = build_model("mobilenet_v3_small", 5, freeze_backbone=True, pretrained=False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 1024 * 5 + 5    # 마지막 분류층만 (04-1의 5,125개)


def test_data_version_reads_dvc_lock(tmp_path):
    lock = tmp_path / "dvc.lock"
    lock.write_text(
        "schema: '2.0'\nstages:\n"
        "  validate:\n    deps:\n    - path: data/raw\n      md5: aaa.dir\n"
        "    outs:\n    - path: data/interim/validation.json\n      md5: bbb\n"
        "  prepare:\n    outs:\n    - path: data/prepared\n      md5: ccc.dir\n")
    assert data_version(str(lock)) == {"data_raw_md5": "aaa.dir", "data_prepared_md5": "ccc.dir"}
