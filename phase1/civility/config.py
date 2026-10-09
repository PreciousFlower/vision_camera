"""配置加载 —— settings.yaml + behaviors.yaml。

设计要点：所有可调参数都在 YAML 里，代码不写死任何业务阈值。
换一个场景（比如从"办公室"换到"食堂"）只改 YAML，不改代码。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ⚠️ 刻意不在模块顶层 import yaml。
# 集群侧的 run_cluster_vlm.py 会 import civility.prompts（为了共用同一份 Prompt），
# 那条路径会连带 import 本模块；若顶层就依赖 yaml，而集群环境没装 pyyaml，
# 作业会在 30 秒内直接崩溃。惰性导入让"只需要 Prompt 逻辑"的场景零依赖。
# （见 tju/10-cluster-handover.md §3.1 的已装包清单 —— 里面没有 pyyaml）

# 项目根目录 = 本文件的上两级（phase1/）
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Behavior:
    """一条不文明行为的定义。"""

    id: str
    name: str
    definition: str
    evidence_required: str = ""
    counter_examples: list[str] = field(default_factory=list)
    observable_cues: list[str] = field(default_factory=list)
    severity: str = "medium"
    enabled: bool = True

    @property
    def is_usable(self) -> bool:
        """定义是否足够完整到可以交给模型。

        definition 为空的行为不能进 Prompt —— 模型会自由发挥，
        这类"定义缺失"是线上误报的主要来源，宁可启动就报错。
        """
        return bool(self.id and self.name and self.definition.strip())


@dataclass
class BehaviorSet:
    behaviors: list[Behavior]
    defaults: dict[str, Any]

    @property
    def enabled(self) -> list[Behavior]:
        return [b for b in self.behaviors if b.enabled and b.is_usable]

    def by_id(self, bid: str) -> Behavior | None:
        for b in self.behaviors:
            if b.id == bid:
                return b
        return None


def _read_yaml(path: Path) -> dict:
    import yaml  # 惰性导入，见文件头说明

    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_behaviors(path: str | Path | None = None) -> BehaviorSet:
    p = Path(path) if path else ROOT / "config" / "behaviors.yaml"
    raw = _read_yaml(p)
    items = []
    for b in raw.get("behaviors", []):
        items.append(
            Behavior(
                id=str(b.get("id", "")).strip(),
                name=str(b.get("name", "")).strip(),
                definition=str(b.get("definition", "") or "").strip(),
                evidence_required=str(b.get("evidence_required", "") or "").strip(),
                counter_examples=[str(x) for x in (b.get("counter_examples") or [])],
                observable_cues=[str(x) for x in (b.get("observable_cues") or [])],
                severity=str(b.get("severity", "medium")),
                enabled=bool(b.get("enabled", True)),
            )
        )

    # 唯一性检查：id 重复会让告警统计串味，必须在启动时炸掉
    seen = set()
    for b in items:
        if b.id in seen:
            raise ValueError(f"behaviors.yaml 里 id 重复: {b.id}")
        seen.add(b.id)

    return BehaviorSet(behaviors=items, defaults=raw.get("defaults") or {})


def load_settings(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "config" / "settings.yaml"
    cfg = _read_yaml(p)

    # 相对路径统一解析到项目根，避免"从别的目录启动就找不到数据"
    for key in ("data_dir", "frames_dir", "results_dir"):
        v = (cfg.get("paths") or {}).get(key)
        if v and not Path(v).is_absolute():
            cfg["paths"][key] = str(ROOT / v)

    bh = (cfg.get("paths") or {}).get("behaviors")
    if bh and not Path(bh).is_absolute():
        cfg["paths"]["behaviors"] = str(ROOT / bh)

    return cfg


def get_api_key(cfg: dict) -> str | None:
    """从环境变量取 API Key。绝不写入配置文件、绝不入库。"""
    env_name = (cfg.get("cloud") or {}).get("api_key_env", "DASHSCOPE_API_KEY")
    for name in (env_name, "SKYWORK_API_KEY", "OPENAI_API_KEY"):
        v = os.environ.get(name)
        if v:
            return v.strip()
    return None
