"""Prompt 编译器 —— 把 behaviors.yaml 里的行为定义变成可执行的 Prompt。

这是第一阶段的核心资产。三个关键设计：

1. **结构化输出 + Schema 校验**
   要求模型只输出一个 JSON 对象。工程上必须配套一个校验器，
   不符合 Schema 的输出自动重试，而不是让脏数据流进告警链路。

2. **先描述、后判断（observable 字段）**
   强制模型先写"我在画面上看到的客观事实"，再给出 violation 结论。
   这一步显著降低幻觉率——模型编不出可观察细节时会倾向于报 false。

3. **few-shot 覆盖"负例"**
   两个示例：一个正例（有行为）、一个负例（看起来像但没有）。
   负例示例是压制误报最有效的手段，比在定义里多写三条正例管用。

云端 API 和集群本地模型共用同一份 Prompt（见 build_messages），
保证两边跑出来的结果可比。
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from .config import Behavior, BehaviorSet

# 模型有时会把 JSON 包在 ```json ... ``` 里，或者加解释性前后缀。
# 只用"找到第一个平衡的花括号块"这一种策略，够用且不会误伤。
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_RISK_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SEVERITIES = {"low", "medium", "high"}
_NO_EVIDENCE = {"无", "无证据", "暂无", "未知", "证据不足", "无法判断", "none", "null"}


def _confidence(value: Any) -> float:
    """置信度必须是有限的 0~1 数值，不能把布尔值或 NaN 当作高置信度。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return 0.0
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else 0.0


def _risk_problems(item: Any) -> list[str]:
    """开放风险必须有可复核的结构；非法项不得进入告警候选。"""
    if not isinstance(item, dict):
        return ["risk 元素不是对象"]
    errors: list[str] = []
    rid = item.get("risk_id")
    if not isinstance(rid, str) or not _RISK_ID_RE.fullmatch(rid.strip().lower()):
        errors.append("risk_id 必须是稳定的英文类别 ID")
    for key in ("name", "observable"):
        value = item.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"risk {key} 缺少有效文字")
    evidence = item.get("observable")
    if isinstance(evidence, str) and evidence.strip().lower() in _NO_EVIDENCE:
        errors.append("risk observable 没有可见证据")
    severity = item.get("severity")
    if not isinstance(severity, str) or severity.strip().lower() not in _SEVERITIES:
        errors.append("risk severity 必须是 low/medium/high")
    confidence = item.get("confidence")
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not 0.0 <= confidence <= 1.0 or not math.isfinite(float(confidence))):
        errors.append("risk confidence 必须是有限的 0~1 数值")
    if not isinstance(item.get("alert"), bool):
        errors.append("risk alert 必须是布尔值")
    if item.get("alert") is True:
        reason = item.get("alert_reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append("risk alert=true 时必须给出告警理由")
    return errors


def _schema_for(behaviors: list[Behavior], with_alert: bool = False,
                with_actions: bool = False, with_risks: bool = False) -> dict:
    """生成输出 JSON Schema。行为 id 用 enum 锁定，模型不能自己发明标签。

    with_alert=True   额外要求 alert / alert_reason（aggregate.mode: ai）
    with_actions=True 额外要求 people 数组 —— 描述画面里"每个人在做什么"。
                      这是「摄像头监测」页的核心：不只看有没有违规，
                      还要说清此刻镜头前的人在干什么、动作有无变化。
    with_risks=True   额外要求独立 risks 数组，类别可由模型根据可见证据扩展。
    """
    ids = [b.id for b in behaviors]
    # ⚠️ 字段表就是"输出预算表"：实测 35.8 token/s ≈ 每 token 28 毫秒，
    #    多一个字段就要多写键名+值，3 个行为就是 3 倍。
    #    所以这里**只保留会被使用的字段**：
    #      · person_desc / bbox —— 已从 schema 移除。
    #        它们从没被页面用过（页面用的是 people 数组里的位置和动作），
    #        但模型每次仍会老老实实写 "person_desc": ""、"bbox": []，
    #        三个行为加起来白花几十个 token。sanitize 会自动补默认值，不影响下游。
    item_props: dict[str, Any] = {
        "behavior_id": {"type": "string", **({"enum": ids} if ids else {})},
        "observable": {
            "type": "string",
            "description": "violation=true 时必须填写：按帧序说明开始、关键动作、结果，≤120字，不可补全遮挡部分",
        },
    }
    if any(b.evidence_checks for b in behaviors):
        item_props["evidence"] = {
            "type": "object",
            "description": "命中时按该行为 evidence_checks 逐项引用本窗口帧序号（从1开始）；看不清填0并判false",
            "additionalProperties": {"type": "integer", "minimum": 0},
        }
    item_props.update({"violation": {"type": "boolean"},
                       "confidence": {"type": "number", "minimum": 0, "maximum": 1,
                                      "description": "violation=true 时才填，0~1"}})
    item_required = ["behavior_id", "violation"]

    if with_alert:
        item_props["alert"] = {
            "type": "boolean",
            "description": "该行为此刻是否已经足以触发告警（由你判断）",
        }
        item_props["alert_reason"] = {
            "type": "string",
            "description": "**仅 alert=true 时填**：一句话理由（≤48 字）；"
                           "alert=false 时**省略该键**",
        }
        item_required += ["alert"]

    props: dict[str, Any] = {
        "scene": {
            "type": "string",
            "description": "一句话描述画面场景（谁、在哪、在做什么）",
        },
        "violations": {
            "type": "array",
            **({"maxItems": 0} if not ids else {}),
            "items": {
                "type": "object",
                "required": item_required,
                "properties": item_props,
            },
        },
    }
    required = ["violations", "scene"]

    if with_risks:
        props["risks"] = {
            "type": "array", "maxItems": 16,
            "description": "配置行为以外有可见证据的现场风险；没有风险填 []",
            "items": {
                "type": "object",
                "required": ["risk_id", "name", "severity", "confidence", "observable", "alert"],
                "properties": {
                    "risk_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$",
                                "description": "稳定的英文类别，如 fall/fire/conflict/climbing/obstruction；可自定义"},
                    "name": {"type": "string", "description": "具体的中文风险名称"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "observable": {"type": "string", "minLength": 1,
                                   "description": "画面中真实可见的风险证据，说明位置和动作或变化"},
                    "alert": {"type": "boolean"},
                    "alert_reason": {"type": "string",
                                     "description": "alert=true 时必须填写为何现在需要提醒"},
                    "person_desc": {"type": "string", "description": "可选的涉事人物位置描述"},
                },
            },
        }
        required.append("risks")

    if with_actions:
        # ⚠️ 这里用**短字段名 + 短值**是有原因的，不是随手起名：
        #    实测解码速度 35.8 token/s，也就是**每多输出一个 token 多花 28 毫秒**。
        #    原先每个人要写 where/appearance/action/changed 四个长字段、
        #    动作描述还允许 20 字 —— 画面里 2~3 个人的输出就上百 token，
        #    光这一项就把延迟从 5.5 秒推到 11 秒（实测）。
        #    改成 at/act/chg 三个短键、值更短，仍保留"每个人在做什么"这个核心信息。
        #    （sanitize() 会把短键归一成长键，前端与聚合逻辑完全不用改。）
        props["people"] = {
            "type": "array",
            "description": "画面中每个可辨认的人及其当前动作；画面里没人时填空数组",
            "items": {
                "type": "object",
                "required": ["at", "act", "chg"],
                "properties": {
                    "at": {"type": "string",
                           "description": "位置，≤8 字，如「左侧近处」"},
                    "act": {"type": "string",
                            "description": "当前动作，≤32 字，保留物品与关键动作，如「蹲下捡纸杯」"},
                    "chg": {"type": "string",
                            "description": "整个窗口内的动作过程，≤96 字，按时间顺序说明开始、变化和结果；无变化写无明显变化"},
                },
            },
        }
        required.append("people")

    return {"type": "object", "required": required, "properties": props}


def _render_behaviors(behaviors: list[Behavior]) -> str:
    blocks = []
    for b in behaviors:
        lines = [f"### [{b.id}] {b.name}（严重程度：{b.severity}）", f"定义：{b.definition}"]
        if b.evidence_required:
            lines.append(f"判定所需的证据：{b.evidence_required}")
        if b.evidence_checks:
            lines.append("必须逐项核对的 evidence_checks（输出 evidence 中每个 ID 对应一个可见帧序号）：")
            lines += [f"  - {key}: {desc}" for key, desc in b.evidence_checks.items()]
        if b.evidence_sequence:
            lines.append("证据时序：" + " → ".join(b.evidence_sequence)
                         + "；帧号不能倒退，首项必须早于末项，必须是同一人和同一物品。")
        if b.observable_cues:
            lines.append("可观察线索：")
            lines += [f"  - {c}" for c in b.observable_cues]
        if b.counter_examples:
            # 反例单独成段并强调，这是压制误报的主力
            lines.append("以下情况【不算】该行为（误报高发区，务必逐条排除）：")
            lines += [f"  - {c}" for c in b.counter_examples]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_system_prompt(bs: BehaviorSet, with_alert: bool = False,
                        with_actions: bool = False, with_risks: bool = False) -> str:
    behaviors = bs.enabled
    if not behaviors and not with_risks:
        raise ValueError("没有启用任何行为定义，检查 behaviors.yaml 的 enabled 字段")
    ids = ", ".join(f'"{b.id}"' for b in behaviors)
    schema = json.dumps(_schema_for(behaviors, with_alert, with_actions, with_risks),
                        ensure_ascii=False, indent=2)

    # with_actions=True 时追加"描述每个人在做什么"的任务。
    # 这是「摄像头监测」页要的东西：用户想知道的不是"有没有违规"，
    # 而是"镜头前的人在干什么"。所以这里要的是**动作描述**，
    # 而不是又一次的是/否判断。
    action_block = ""
    if with_actions:
        action_block = """
# 动作描述（本次任务的重要部分）
除了逐项检查违规，还要用 `people` 数组**描述画面里每个人此刻在做什么**：
- 画面中每个人一项，用三个**短键**：位置 `at`（≤8 字）、
  当前动作 `act`（≤32 字）、窗口内的动作过程 `chg`（≤96 字）。三个键均须提供。
- 用足够的文字写完整关键过程，保留物品、动作、落点与结果。
  不要为了压短输出只描述最后一帧；不必写与行为无关的外观。
- `act` 要具体到动作本身，避免「站着」「在活动」这类空话：
  · 好：`{"at":"左侧近处","act":"蹲下捡纸杯","chg":"由站立变蹲下"}`
  · 差：`{"at":"画面左侧靠近垃圾桶的位置","act":"一个人正在做某些动作"}`
- `chg` 是**时序信息的关键**：对比这几帧，说明动作怎么变了；
  按时间先后覆盖开始、关键变化和结果，不要只写最后一帧状态。
  持物→脱手→落地、抬手→吸食等关键变化发生时必须写 chg，不能省略。
  没有变化就写「无明显变化」，不能编造变化。
- 画面里没人时，`people` 填空数组 `[]`，不要编造。
- 人很多看不清时，只描述最清楚的 1-3 个，其余不必强行编。
- 遮挡时直说「手遮嘴」「手持不明物」，看清指甲剪就描述指甲剪；不要补成「吸烟」。
- 「提着垃圾袋」只能描述为持物；没有看到脱手、落地，就不能在 act/chg/scene 里编写这些动作。
"""

    # with_alert=True 时追加一段"由你决定要不要告警"的指令。
    # 这不是简单的"把 violation 复制到 alert" —— 而是明确要求模型区分
    # 「发生了」和「值得打扰人」两件事：
    #   只有行为本身已经确认，才允许 violation=true；看不清时必须 false。
    # 这正是让 AI 判断的价值所在，也是最容易被实现成"两个字段永远相等"的地方。
    alert_block = ""
    if with_alert:
        alert_block = """
# 告警判定（本次任务的重要部分）
每个行为除了判断"有没有发生"，还要判断**此刻是否已经值得告警**：
- `alert` 字段填 true / false，`alert_reason` 用一句话说明理由。
- **violation 与 alert 不是同一件事**，请分开判断：
  · `violation=true, alert=false` —— 必要证据全部可见，确实发生过，但此刻无需提醒。
    例如：人员已捡回刚丢下的垃圾，现场风险已经解除。
    完整丢弃已发生且垃圾仍留在地面，即使人物已离开也应告警，便于留存事件和清理。
  · 疑似烟雾、人物太小、物体不可辨认或关键动作完全被遮住：`violation=false, alert=false`。
    正常持烟时手部局部遮嘴，若烟支和使用动作仍清晰，不应仅因此要求复核。
  · `violation=false, alert=true` —— 不应出现这种组合。
  · `violation=true, alert=true` —— 行为清楚、动作链条完整，现在就该提醒。
- 判 alert=true 的门槛：**你愿意仅凭当前这几帧就去提醒现场的人**吗？
  愿意才填 true。犹豫就填 false，并说明还缺什么证据。
- 不要为了"多报点"而放宽：误报会让人关掉系统，代价高于漏报。
"""

    risk_block = ""
    if with_risks:
        risk_block = """
# 开放风险识别（不限预定义行为）
除配置行为外，主动观察其他需要现场人员关注的安全风险或不文明行为，用独立 `risks` 数组报告。
- 类别不限于吸烟、扔垃圾：如摔倒、火情、打斗、危险攀爬、通道堵塞；可根据现场自定义类别。
- `risk_id` 是稳定的英文类别 ID（如 fall/fire/conflict/climbing/obstruction），
  同类风险在不同窗口使用相同 ID，不要加入时间、人物或随机编号。`name` 写具体的中文名称。
- 已由启用行为覆盖的同一件事只写入 `violations`，不要在 `risks` 重复报告。
- 配置行为因证据不足判 false 时，不得换成「疑似吸烟」「烟雾风险」「垃圾落地」放进 risks 绕过证据要求。
- 每条风险必须给出 `severity`（low/medium/high）、0~1 的 `confidence`、
  真实可见的 `observable` 证据、布尔值 `alert`；`alert=true` 还必须写 `alert_reason`。
- 只有证据清晰、当前值得提醒现场人员时 `alert=true`；可疑但证据不足时不告警。
  不要根据衣着、身份、情绪或动机推断危险，不要把普通行走、交谈、弯腰捡物当作风险。
- 火焰、烟雾、通道堵塞等环境风险可在无人时报告；需要人的行为必须看得见涉事人物。
- 不存在有可见证据的额外风险时填 `risks:[]`，不要为了填满数组编造。
"""

    # 长度纪律里那两处提到 alert 的地方必须按开关生成 ——
    # 关掉告警判定时还提 alert_reason，模型就会凭空多输出一个没人要的字段
    # （实测被测试用例抓到："非 ai 模式不该出现 alert_reason"）。
    alert_key_hint = " / `alert_reason`" if with_alert else ""
    alert_line = ("\n  `alert_reason` 只在 `alert=true` 时写。" if with_alert else "")
    negative_keys = "`behavior_id`、`violation` 和 `alert=false`" if with_alert else "`behavior_id` 和 `violation`"

    # 「一行写完」的字面示例。
    # ⚠️ 用**纯字符串拼接**拼出来，不要用 f-string、也不要用 .format()：
    #    JSON 里的花括号和这两种语法都会打架（f-string 会把 `{"scene"...}` 当成
    #    格式占位符直接抛 ValueError；.format() 则要求把每个 { 写成 {{，
    #    一旦拼接顺序没注意就会漏掉转义，输出成 {{"at":...}} 这样的坏示例）。
    #    这个坑我两种都踩过一遍，所以这里最简单直白地拼。
    _ex_people = (',"people":[{"at":"左侧近处","act":"蹲下捡纸杯","chg":"由站立变蹲下，捡起地面纸杯"}]'
                  if with_actions else "")
    example_items = []
    for behavior in behaviors:
        example_items.append('{"behavior_id":"' + behavior.id + '","violation":false'
                             + (',"alert":false' if with_alert else '') + '}')
    compact_example = ('{"scene":"一人蹲下捡纸杯","violations":[' + ','.join(example_items)
                       + ']' + _ex_people + (',"risks":[]' if with_risks else '') + '}')

    return f"""你是一个公共场所视频行为分析助手。你会看到同一时间窗口内的若干连续视频帧（按时间先后排列）。

# 任务
逐一检查下列行为是否发生，并只依据画面中**可见的证据**判断。
{alert_block}{action_block}{risk_block}
# 行为定义
{_render_behaviors(behaviors) or '未启用配置行为，violations 填空数组 []，继续识别开放风险。'}

# 判定纪律（必须遵守）
1. **先描述，再判断**：`observable` 只写画面上真实可见的事实，不写推测。
2. **证据不足即判 false**：宁可漏报，不可凭"看起来像"就报 true。监控场景里
   误报的代价（打扰、失去信任）通常高于漏报。
3. **逐条排除反例**：对每个行为，先想一遍"反例清单里有没有命中的"，命中就判 false。
4. **只报你确实看到的**：不要因为"这个场景通常会有违规"而推断存在违规。
5. **人不在画面里就不报**：没有可辨认的人时，所有需要人的行为一律 false。
6. **不要把地面上的物体当成人**：只看可辨认的人形轮廓。
7. **不可用常识补全遮挡或漏采的过程**：看不清手中物、口部或物体落点，就是未知。
   指甲剪、钥匙、反光不能当烟支；手遮嘴不能推出吸烟，更不能推出烟雾。
   提着垃圾袋不能推出丢弃；手变空或物体出画不能推出垃圾落地。
8. **证据必须能回看**：对定义含 evidence_checks 的行为，判 true 前先写 observable 和 evidence。
   evidence 的键必须覆盖该行为全部检查项，值为实际输入帧序号（第一张=1，与文件名无关）。
   任一项未看到或有反例就判 false，不得编帧号。一个帧号只是复核索引，不是事实保证。
   scene、people、observable、risks 均遵守同一证据纪律，不能在文字里把未知写成确定发生。

# 输出长度与事实完整性
- ⚠️ **必须输出紧凑的单行 JSON**：不要换行、不要缩进、不要多余空格。
  美化输出（每个字段一行、缩进两格）会凭空多出 30%~50% 的 token，
  而 token 就是用户的等待时间 —— 实测每 token 约 28 毫秒。
  即整个对象在一行里写完，键值之间不加空格。
- 未命中（`violation=false`）的行为：**只写 {negative_keys}**，
  其余键（`confidence` / `observable`{alert_key_hint}）**全部省略**。
  但**判断过程不能省** —— 你必须先逐条排除反例再下结论，只是不必写出来。
- 命中（`violation=true`）的行为：必须写 `confidence` 与 `observable`
  （≤120 字），按实际帧序写清开始→关键动作→结果，可引用对应帧号；
  必要证据不可为缩短输出而省略。定义有检查项时还必须写 `evidence`。{alert_line}
- `scene` 一句话，不超过48字，简述实际观察到的过程。
- 除 JSON 外不要输出任何文字。事实准确、动作过程完整优先，避免无关赘述。
- ⚠️ 命中的行为**必须**有 `observable`：它是复核与排查误报的唯一依据，
  没有它，人无法验证你的结论。空依据的 true 判定等于没有结论。

# 输出格式
⚠️ **一行写完，不许换行、不许缩进、不许对齐**。整个 JSON 必须是单个行：
以左花括号开头、右花括号结尾，键值之间用逗号分隔，冒号后不加空格。
美化输出（每个字段单独一行）会凭空多出三到五成 token，而 token 就是用户
的等待时间（实测每 token 约 28 毫秒）。

只输出这一个 JSON 对象，不要任何其他文字、解释或 Markdown 代码块标记。
`violations` 必须包含全部 {len(behaviors)} 个行为，顺序固定：
{ids}

正确形式的示例（注意：全部在同一行）：
{compact_example}

JSON Schema：
{schema}
"""


def _few_shot(bs: BehaviorSet, with_actions: bool = False,
              with_risks: bool = False, with_alert: bool = False) -> list[tuple[str, str]]:
    """返回 [(用户提问, 模型应答)] 示例。

    用明确标记的教学情境对照正例和相似负例；它们不是当前窗口的事实。
    示例里的 behavior_id 动态取自实际启用的行为，避免出现不存在的 id。
    """
    behaviors = bs.enabled
    ids = [b.id for b in behaviors]

    def skeleton() -> list[dict]:
        # 未命中只写标识与否定判断（AI 告警模式还包含 alert=false）。
        # 示例必须和规则完全一致，
        # 否则 few-shot 会教坏模型（踩过一次：示例带 observable、规则说可省略，
        # 结果模型跟着示例写长输出）
        return [{"behavior_id": i, "violation": False,
                 **({"alert": False} if with_alert else {})} for i in ids]

    # ── 负例：地上有垃圾，但没人扔 ──
    # ⚠️ 这个示例的 observable **必须留空**，因为指令就是这么要求的
    #   （"判 false 时留空 observable"，为压延迟）。
    #   踩过的坑：第一版指令说"必须填写"、示例却留空，模型跟着示例走，
    #   结果所有行为都不写依据 —— few-shot 的示范作用永远大于文字规定，
    #   所以两者必须一致。现在指令也改成留空，两边对齐了。
    neg = skeleton()
    neg_scene = "空无一人的走廊，地面散落纸屑与空瓶。"

    if with_actions:
        neg_people: list[dict] = []
    else:
        neg_people = None

    pre = "以下是假设的教学场景，不是当前视频的事实；只按给定事实输出 JSON。"

    def _doc(scene: str, viol: list[dict], people: list[dict] | None) -> str:
        obj: dict = {"scene": scene, "violations": viol}
        if people is not None:
            obj["people"] = people
        if with_risks:
            obj["risks"] = []
        # separators 去掉所有多余空格 —— 示例是模型模仿的对象，
        # 示例紧凑，模型才会紧凑（这几字节在真实输出里会放大成几十个 token）
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    examples = [
        (f"{pre}\n\n（示例A：负例 —— 有垃圾但无人丢弃）",
         _doc(neg_scene, neg, neg_people)),
        (f"{pre}\n\n一人拿指甲剪靠近嘴巴，手遮挡口部；未见烟支、未见烟雾。",
         _doc("一人手持指甲剪遮嘴", skeleton(),
              [{"at": "中央", "act": "手持指甲剪遮嘴", "chg": "工具靠近嘴巴，口部被手遮住，未见烟支"}] if with_actions else None)),
        (f"{pre}\n\n一人提着垃圾袋来回晃动，袋子始终在手上，未见脱手或落地。",
         _doc("一人提着垃圾袋", skeleton(),
              [{"at": "中央", "act": "提着垃圾袋", "chg": "提袋来回晃动，袋子始终未脱手"}] if with_actions else None)),
    ]
    if with_risks:
        risk_doc = {"scene": "门口一人由站立摔倒", "violations": skeleton(),
                    "risks": [{"risk_id": "fall", "name": "人员摔倒", "severity": "high",
                               "confidence": 0.92, "observable": "中央人物由站立倒地，后续帧未起身",
                               "alert": True, "alert_reason": "持续倒地，需要现场确认"}]}
        if with_actions:
            risk_doc["people"] = [{"at": "中央近处", "act": "倒地未起身", "chg": "由站立变倒地"}]
        examples.append((f"{pre}\n\n（示例C：配置行为未命中，但出现摔倒风险）",
                         json.dumps(risk_doc, ensure_ascii=False, separators=(",", ":"))))
    positive_examples = {
        "smoking": ("第1帧烟支清楚可见，第2帧火焰点燃嘴边烟支，第3帧吸食并呼出烟雾。",
                    "男子点燃香烟并吸食", "烟支可辨，火焰点燃嘴边烟支后吸食",
                    {"tobacco": 1, "use": 2}),
        "littering": ("第1帧手持瓶子，第3帧同一瓶子脱手下落，第4帧该瓶落在地面，非放入容器、非暂放。",
                     "男子丢弃手中瓶子", "持瓶→脱手→同一瓶子落在地面",
                     {"held": 1, "released": 3, "landed": 4}),
        "no_safety_helmet": ("施工区一名工人背面头发和头顶清楚可见，没有帽体覆盖；黄色安全帽拿在手上。",
                            "工人手持安全帽但未佩戴", "施工区工人裸露头顶，安全帽只拿在手上",
                            {"worksite": 1, "bare_head": 1}),
    }
    for bid, (facts, scene, observable, evidence) in positive_examples.items():
        behavior = bs.by_id(bid)
        if not behavior or bid not in ids or not set(behavior.evidence_checks) <= set(evidence):
            continue
        positive = skeleton()
        candidate = {"behavior_id": bid, "observable": observable,
                     "evidence": {k: v for k, v in evidence.items() if k in behavior.evidence_checks},
                     "violation": True, "confidence": 0.9}
        if with_alert:
            candidate.update(alert=True, alert_reason="必要证据明确可见")
        positive[ids.index(bid)] = candidate
        examples.append((f"{pre}\n\n{facts}", _doc(scene, positive,
                         [{"at": "中央", "act": scene, "chg": observable}] if with_actions else None)))
    if "smoking" in ids:
        examples.append((f"{pre}\n\n烟支叼在嘴里，刚拿出打火机；没有火焰、没有吸食或烟雾，尚未开始吸烟。",
                         _doc("男子准备点烟", skeleton(),
                              [{"at": "中央", "act": "准备点烟", "chg": "拿出打火机，未见点燃或吸食"}] if with_actions else None)))
    return examples


def build_messages(
    bs: BehaviorSet,
    frame_images: list,
    frame_labels: list[str] | None = None,
    with_few_shot: bool = True,
    with_alert: bool = False,
    with_actions: bool = False,
    with_risks: bool = False,
) -> list[dict]:
    """构造 chat 消息。云端和集群**共用这一个函数**，保证两边 Prompt 逐字相同。

    frame_images 元素可以是：
        - str            ：路径 / http(s) URL / data URL  → 产出 image_url 部件（云端用）
        - PIL.Image.Image：图片对象                        → 产出 image 部件（集群用）

    为什么不给两个后端各写一套：只要 Prompt 有一处不一致，
    云端和集群的准确率数字就没有可比性 —— 而"两种后端的结果可比"
    正是第一阶段选型的全部依据。

    with_alert=True  ：要求模型自己判断该不该告警（aggregate.mode: ai）
    with_actions=True：要求模型描述画面里每个人在做什么（摄像头监测页）
    with_risks=True  ：识别配置以外的开放风险；允许配置行为为空
    """
    messages: list[dict] = [
        {"role": "system",
         "content": build_system_prompt(bs, with_alert=with_alert,
                                        with_actions=with_actions, with_risks=with_risks)}
    ]

    if with_few_shot:
        for user_text, assistant_text in _few_shot(bs, with_actions=with_actions,
                                               with_risks=with_risks, with_alert=with_alert):
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})

    labels = frame_labels or [f"帧{i + 1}" for i in range(len(frame_images))]
    if len(labels) != len(frame_images):
        raise ValueError("帧标签数量必须与图片数量一致，不能静默丢帧")
    content: list[dict] = [
        {
            "type": "text",
            "text": "以下是按时间先后排列的视频抽样帧，时间间隔可能不等："
            + "、".join(labels)
            + "。请核对每帧时间，不得推断未呈现的中间过程，"
              "不得把早期画面中的动作当成末帧正在发生的动作。请按要求输出 JSON。",
        }
    ]

    for index, (label, img) in enumerate(zip(labels, frame_images), 1):
        content.append({"type": "text", "text": f"[帧{index}；{label}]"})
        if isinstance(img, str):
            content.append({"type": "image_url", "image_url": {"url": img}})
        else:
            # PIL.Image：qwen-vl-utils 的 process_vision_info 认这个结构
            content.append({"type": "image", "image": img})

    messages.append({"role": "user", "content": content})
    return messages


# ─────────────────────────── 输出解析与校验 ───────────────────────────


def extract_json(text: str) -> dict | None:
    """从模型输出里抠出 JSON 对象。容忍代码块包裹和前后废话。"""
    if not text:
        return None
    candidates: list[str] = []

    m = _FENCE_RE.search(text)
    if m:
        candidates.append(m.group(1).strip())

    start = text.find("{")
    if start >= 0:
        # 找与之平衡的右花括号（忽略字符串内的括号）
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break

    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def validate(obj: dict, bs: BehaviorSet, *, strict: bool = True,
             with_risks: bool = False) -> tuple[bool, list[str]]:
    """校验模型输出是否符合 Schema。返回 (是否合法, 问题列表)。

    校验失败 ⇒ 上层重试。这是让"自定义行为"能安全扩展的关键：
    用户往 YAML 里加行为后，模型偶尔会漏项或造 id，必须挡住。
    with_risks=True 的新请求必须返回 risks（允许 []）；旧请求与历史结果可不带。

    ⚠️ strict 的含义（踩过一次）：
        **结构问题**（漏项、造 id、类型不对）→ 必须重试，strict 与否都算错误。
        **质量 lint**（比如所有 observable 都为空）→ 只是提示，**不应触发重试**。

        最初把 lint 也算成错误，结果"模型判定所有行为都不存在、所以不写描述"
        这种**完全合法**的输出会被判非法 → 无限重试、白烧机时。
        现在 lint 默认只在 strict=True 时报出（供自检与调试用），
        在线路径（云端/集群）传 strict=False，只把它当日志提示。
    """
    problems: list[str] = []
    lint: list[str] = []
    if not isinstance(obj, dict):
        return False, ["输出不是 JSON 对象"]

    expected = [b.id for b in bs.enabled]
    if "violations" not in obj or not isinstance(obj["violations"], (list, dict)):
        return False, ["缺少 violations 数组/对象"]

    # 兼容两种形状：list（模型原始输出）与 dict（已 sanitize 过的结果）。
    # ⚠️ dict 形状的**值里没有 behavior_id**（id 是键），必须补进去，
    #    否则下面的 bid 取到 None，会被判成"未知行为 + 缺少行为项"。
    #    这正是真实链路（webui/server.py 传 dict 给聚合）会走的分支。
    raw = obj["violations"]
    if isinstance(raw, dict):
        items = []
        for bid_key, item in raw.items():
            if isinstance(item, dict):
                merged = dict(item)
                merged.setdefault("behavior_id", bid_key)
                items.append(merged)
            else:
                items.append(item)
    else:
        items = raw

    got = {}
    for item in items:
        if not isinstance(item, dict):
            problems.append("violations 元素不是对象")
            continue
        bid = item.get("behavior_id")
        if bid not in expected:
            problems.append(f"未知 behavior_id: {bid!r}")
            continue
        if not isinstance(item.get("violation"), bool):
            problems.append(f"{bid}: violation 不是布尔值")
            continue
        got[bid] = item

    missing = [i for i in expected if i not in got]
    if missing:
        problems.append(f"缺少行为项: {missing}")

    if "scene" not in obj:
        problems.append("缺少 scene 字段")

    # 请求主动开启时不能悄悄漏掉风险识别；旧请求与历史结果仍兼容。
    if with_risks and "risks" not in obj:
        problems.append("请求开启了开放风险识别，但输出缺少 risks 字段")
    # 新协议可选：旧结果不带 risks 仍合法；带了就必须满足安全边界。
    if "risks" in obj:
        risks = obj["risks"]
        if not isinstance(risks, list):
            problems.append("risks 不是数组")
        else:
            if len(risks) > 16:
                problems.append("risks 超过 16 项")
            for index, risk in enumerate(risks):
                problems.extend(f"risks[{index}]: {p}" for p in _risk_problems(risk))

    # ── 以下为质量 lint：不触发重试，只提示 ──
    # ⚠️ 这条 lint 随 Prompt 一起改过，别照老版本理解：
    #    老 Prompt 要求"判 false 也必须写 observable"，那时"全空"就是退化。
    #    现在为了压延迟（每 token 约 28 毫秒）改成"判 false 时留空"，
    #    所以"全空"在没人没违规时是**正常**的，不能再当异常报。
    #    真正该抓的退化成了一条更精确的规则：**判 true 却没给依据** ——
    #    那种结论无法复核、误报也无从排查，必须提示。
    if isinstance(obj.get("scene", ""), str) and not obj.get("scene", "").strip():
        lint.append("scene 为空 —— 模型可能把'简短'误解成'省略论证'")
    hits_without_evidence = [
        bid for bid, item in got.items()
        if item.get("violation")
        and not str(item.get("observable", "") or "").strip()
    ]
    if hits_without_evidence:
        lint.append(
            f"命中的行为没有写 observable：{hits_without_evidence} —— "
            f"判 true 却给不出可见依据，人无法复核"
        )

    if strict:
        problems += lint
    return (len(problems) == 0), problems


def _evidence_issues(item: dict, behavior: Behavior, frame_count: int | None) -> list[str]:
    """Check evidence references, never claim to independently verify image pixels.

    Incomplete positive output is suppressed, not retried until the model invents evidence.
    """
    issues: list[str] = []
    observable = item.get("observable")
    if (not isinstance(observable, str) or not observable.strip()
            or observable.strip().lower() in _NO_EVIDENCE):
        issues.append("缺少可见事实说明")
    if not behavior.evidence_checks:
        return issues
    evidence = item.get("evidence")
    if not isinstance(evidence, dict):
        return issues + ["缺少逐项帧证据（旧服务需重新部署并重启）"]
    for key, description in behavior.evidence_checks.items():
        ref = evidence.get(key)
        if (type(ref) is not int or ref < 1
                or (frame_count is not None and ref > frame_count)):
            issues.append(f"{key}：缺少有效帧证据（{description}）")
    if not issues and behavior.evidence_sequence:
        refs = [evidence[key] for key in behavior.evidence_sequence]
        if refs[0] >= refs[-1] or refs != sorted(refs):
            issues.append("动作链条的帧序号不成立：需按先后顺序且跨越不同帧")
    return issues


def sanitize(obj: dict, bs: BehaviorSet, *, frame_count: int | None = None) -> dict:
    """把输出整理成统一内部结构，缺失项补成 false 而不是丢弃。

    补 false 而不是补 true —— 缺项按"未发现"处理，符合"证据不足即不报"的纪律。

    ⚠️ 必须同时接受两种 `violations` 形状（踩过一次严重 bug）：
        list 形式：[{"behavior_id": "...", "violation": ...}, ...] ← 模型原始输出
        dict 形式：{"littering": {...}, "smoking": {...}}          ← 已 sanitize 过的结果

    历史事故：本函数原只认 list。而 aggregate.derive_events() 会把
    **已 sanitize 过的 dict** 再喂回来一次（因为它同时要处理 mock 后端
    未 sanitize 的原始输出）。结果遍历 dict 拿到的是字符串键，
    `isinstance(item, dict)` 全不成立 → 每个行为都被判成 missing → 默认 False
    → **告警永远为 0 条**。而网页的逐窗口列表直接读原始 dict，显示正常，
    所以表面上"模型明明识别出来了、却没告警"，极难排查。

    这类 bug 的教训：**同一个函数被两条路径调用时，输入契约必须写死并校验**，
    不能靠调用方"记得传对格式"。
    """
    if not isinstance(obj, dict):
        obj = {}
    out: dict[str, Any] = {
        "scene": str(obj.get("scene", "") or ""),
        "violations": {},
        # people：画面里每个人在做什么（仅 with_actions=True 时模型会给）。
        # 归一化成统一结构，前端不必再判断字段缺失。
        "people": [],
        "risks": [],
    }

    people = obj.get("people")
    for p in (people if isinstance(people, list) else []):
        if not isinstance(p, dict):
            continue
        # ⚠️ 两套键名都要认：
        #   短键 at/act/chg —— 现在 Prompt 里要求的紧凑格式（为压输出长度，
        #     实测每 token 约 28 毫秒，短字段直接换延迟）；
        #   长键 where/appearance/action/changed —— 老格式，
        #     云端模型、用户自己改过的 Prompt、以及历史结果都还在用。
        #   归一化在这里做完，前端与聚合逻辑就不用关心是哪种。
        act = str(p.get("act") or p.get("action") or "").strip()
        if not act:
            continue          # 没写动作的条目没有价值，丢掉
        out["people"].append({
            "where": str(p.get("at") or p.get("where") or ""),
            "appearance": str(p.get("appearance") or ""),
            "action": act,
            "changed": str(p.get("chg") or p.get("changed") or ""),
        })

    # 同类别风险只保留一次。非法字段直接舍弃，不靠 truthiness 将字符串变成告警。
    # 更严重的有效告警优先，避免模型在一个数组里重复输出时被低等级覆盖。
    risk_by_id: dict[str, dict] = {}
    raw_risks = obj.get("risks")
    for risk in (raw_risks[:16] if isinstance(raw_risks, list) else []):
        if _risk_problems(risk):
            continue
        risk_id = risk["risk_id"].strip().lower()
        clean_risk = {
            "risk_id": risk_id,
            "name": risk["name"].strip()[:100],
            "severity": risk["severity"].strip().lower(),
            "confidence": _confidence(risk["confidence"]),
            "observable": risk["observable"].strip()[:1000],
            "alert": risk["alert"],
            "alert_reason": (risk.get("alert_reason", "").strip()[:500]
                             if isinstance(risk.get("alert_reason", ""), str) else ""),
            "person_desc": (risk.get("person_desc", "").strip()[:200]
                            if isinstance(risk.get("person_desc", ""), str) else ""),
        }
        # Prevent known configured behaviors from bypassing their evidence gate via risks.
        # This matches identities/common labels only, not arbitrary semantic paraphrases.
        canonical_id = re.sub(r"^(?:suspected_|possible_)", "", risk_id)
        canonical_id = re.sub(r"_risk$", "", canonical_id)
        canonical_name = re.sub(r"^(?:疑似|可能)", "", clean_risk["name"])
        canonical_name = re.sub(r"风险$", "", canonical_name).strip()
        duplicate = next((b for b in bs.enabled
                          if canonical_id == b.id or canonical_name == b.name), None)
        if duplicate:
            clean_risk["alert"] = False
            clean_risk["evidence_status"] = "insufficient"
            clean_risk["evidence_issues"] = [f"应按配置行为「{duplicate.name}」核对证据，不能改名为开放风险告警"]
            clean_risk["configured_behavior_id"] = duplicate.id
        priority = lambda x: (x["alert"], {"low": 0, "medium": 1, "high": 2}[x["severity"]],
                              x["confidence"])
        if risk_id not in risk_by_id or priority(clean_risk) > priority(risk_by_id[risk_id]):
            risk_by_id[risk_id] = clean_risk
    out["risks"] = list(risk_by_id.values())

    raw = obj.get("violations")
    by_id: dict[str, dict] = {}

    if isinstance(raw, dict):
        # 已是 {bid: {...}} 形式，直接用
        for bid, item in raw.items():
            if isinstance(item, dict):
                by_id[str(bid)] = item
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("behavior_id"):
                by_id[str(item["behavior_id"])] = item
    # raw 为 None 或其他类型：全部按缺失处理（下面的循环会补 false）

    for b in bs.enabled:
        item = by_id.get(b.id)
        if not item:
            out["violations"][b.id] = {
                "violation": False,
                "confidence": 0.0,
                "observable": "",
                "person_desc": "",
                "bbox": None,
                "alert": None,
                "alert_reason": "",
                "missing": True,
            }
            continue
        conf = _confidence(item.get("confidence", 0.0))
        issues = (_evidence_issues(item, b, frame_count)
                  if item.get("violation") is True else [])
        if item.get("evidence_status") == "insufficient" and not issues:
            previous = item.get("evidence_issues")
            issues = [s for s in (previous if isinstance(previous, list) else [])
                      if isinstance(s, str)]
        out["violations"][b.id] = {
            "violation": item.get("violation") is True and not issues,
            "confidence": conf,
            "observable": str(item.get("observable", "") or ""),
            "person_desc": str(item.get("person_desc", "") or ""),
            "bbox": item.get("bbox") if isinstance(item.get("bbox"), list) else None,
            # 模型自判的告警决定（aggregate.mode: ai）。
            # alert 缺省时回落成 None（而不是 False）—— "模型没表态" 和
            # "模型明确说不用告警"必须能区分，否则上层没法决定回退策略。
            "alert": (False if issues else item["alert"] if isinstance(item.get("alert"), bool)
                      else (None if item.get("alert") is None else False)),
            "alert_reason": str(item.get("alert_reason", "") or ""),
            "missing": False,
        }
        clean = out["violations"][b.id]
        if b.evidence_checks or issues:
            evidence = item.get("evidence")
            clean["evidence"] = {key: evidence[key] for key in b.evidence_checks
                                 if isinstance(evidence, dict) and type(evidence.get(key)) is int}
            clean["evidence_status"] = ("insufficient" if issues else
                                        "provided" if clean["violation"] else "not_required")
            clean["evidence_issues"] = issues
    out["risks"] = [risk for risk in out["risks"]
                    if not (risk.get("configured_behavior_id")
                            and out["violations"].get(risk["configured_behavior_id"], {}).get("violation"))]
    return out
