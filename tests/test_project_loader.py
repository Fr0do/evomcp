from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evomcp.pipeline.registry import DEFAULT_REGISTRY
from evomcp.project_loader import infer_project_root, load_project_slots
from evomcp.server import tool_evolve_list_slots


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_infer_project_root_from_config_path(tmp_path: Path) -> None:
    project_root = tmp_path / "demo"
    _write(project_root / "optim" / "search_spaces" / "__init__.py", "")
    config_path = project_root / "configs" / "gepa.yaml"
    _write(config_path, "target_text_slots: []\n")

    assert infer_project_root(config_path) == project_root


def test_load_project_slots_imports_project_registry(tmp_path: Path) -> None:
    project_root = tmp_path / "demo"
    _write(project_root / "optim" / "__init__.py", "")
    _write(
        project_root / "optim" / "search_spaces" / "__init__.py",
        "\n".join(
            [
                "from evomcp.pipeline.registry import DEFAULT_REGISTRY, ProgSlot, SlotKind, TextSlot",
                "DEFAULT_REGISTRY.register_text(TextSlot(name='critic_prompt', role='system', seed_value='Be strict'))",
                "DEFAULT_REGISTRY.register_prog(ProgSlot(name='temperature', kind=SlotKind.SCALAR, default=0.2, bounds=(0.0, 1.0)))",
                "",
            ]
        ),
    )
    config_path = project_root / "configs" / "gepa.yaml"
    _write(config_path, "target_text_slots:\n  - critic_prompt\n")

    load_project_slots(config_path)

    assert "critic_prompt" in DEFAULT_REGISTRY.text_slots
    assert "temperature" in DEFAULT_REGISTRY.prog_slots


def test_tool_evolve_list_slots_returns_project_root_and_slots(tmp_path: Path) -> None:
    project_root = tmp_path / "demo"
    _write(project_root / "optim" / "__init__.py", "")
    _write(
        project_root / "optim" / "search_spaces" / "__init__.py",
        "\n".join(
            [
                "from evomcp.pipeline.registry import DEFAULT_REGISTRY, TextSlot",
                "DEFAULT_REGISTRY.register_text(TextSlot(name='judge_prompt', role='system', seed_value='Judge carefully'))",
                "",
            ]
        ),
    )

    result = tool_evolve_list_slots(str(project_root))

    assert result["project_root"] == str(project_root)
    assert "judge_prompt" in result["text_slots"]
