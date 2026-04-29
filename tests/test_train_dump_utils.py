from argparse import Namespace

import torch

from slime.utils.train_dump_utils import save_debug_train_data


def test_save_debug_train_data_uses_policy_name(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    args = Namespace(
        save_debug_train_data=str(tmp_path / "{policy_name}" / "{rollout_id}_{rank}.pt"),
        policy_name="rewriter",
    )

    save_debug_train_data(args, rollout_id=3, rollout_data={"tokens": [[1, 2, 3]]})

    path = tmp_path / "rewriter" / "3_0.pt"
    payload = torch.load(path, weights_only=False)
    assert payload["policy_name"] == "rewriter"
    assert payload["rollout_id"] == 3
    assert payload["rank"] == 0


def test_save_debug_train_data_defaults_policy_name(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    args = Namespace(
        save_debug_train_data=str(tmp_path / "{policy_name}" / "{rollout_id}_{rank}.pt"),
    )

    save_debug_train_data(args, rollout_id=4, rollout_data={})

    payload = torch.load(tmp_path / "default" / "4_1.pt", weights_only=False)
    assert payload["policy_name"] == "default"
