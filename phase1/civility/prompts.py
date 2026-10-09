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
import re
from typing import Any

from .config import Behavior, BehaviorSet

# 模型有时会把 JSON 包在 ```json ... ``` 里，或者加解释性前后缀。
# 只用"找到第一个平衡的花括号块"这一种策略，够用且不会误伤。
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _schema_for(behaviors: list[Behavior], with_alert: bool = False,
                with_actions: bool = False) -> dict:
    """生成输出 JSON Schema。行为 id 用 enum 锁定，模型不能自己发明标签。

    with_alert=True   额外要求 alert / alert_reason（aggregate.mode: ai）
    with_actions=True 额外要求 people 数组 —— 描述画面里"每个人在做什么"。
                      这是「摄像头监测」页的核心：不只看有没有违规，
                      还要说清此刻镜头前的人在干什么、动作有无变化。
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
        "behavior_id": {"type": "string", "enum": ids},
        "violation": {"type": "boolean"},
        "confidence": {"type": "number",
                       "description": "violation=true 时才填，0~1"},
        "observable": {
            "type": "string",
            "description": "仅 violation=true 时填：支撑结论的画面客观事实，≤25 字",
        },
    }
    item_required = ["behavior_id", "violation"]

    if with_alert:
        item_props["alert"] = {
            "type": "boolean",
            "description": "该行为此刻是否已经足以触发告警（由你判断）",
        }
        item_props["alert_reason"] = {
            "type": "string",
            "description": "**仅 alert=true 时填**：一句话理由（≤20 字）；"
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
            "items": {
                "type": "object",
                "required": item_required,
                "properties": item_props,
            },
        },
    }
    required = ["violations", "scene"]

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
                "required": ["at", "act"],
                "properties": {
                    "at": {"type": "string",
                           "description": "位置，≤8 字，如「左侧近处」"},
                    "act": {"type": "string",
                            "description": "当前动作，≤14 字，如「蹲下捡纸杯」"},
                    "chg": {"type": "string",
                            "description": "相对前一帧的变化，≤10 字，如「由站立变蹲下」；无变化可省略"},
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
                        with_actions: bool = False) -> str:
    behaviors = bs.enabled
    if not behaviors:
        raise ValueError("没有启用任何行为定义，检查 behaviors.yaml 的 enabled 字段")
    ids = ", ".join(f'"{b.id}"' for b in behaviors)
    schema = json.dumps(_schema_for(behaviors, with_alert, with_actions),
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
  当前动作 `act`（≤14 字）、相对前一帧的变化 `chg`（≤10 字，无变化可省略）。
- 字段名和内容都要短 —— 输出速度直接决定这条流能不能"实时"，
  每个字都要花时间。**不要**写外观、衣着这类与行为判断无关的信息。
- `act` 要具体到动作本身，避免「站着」「在活动」这类空话：
  · 好：`{"at":"左侧近处","act":"蹲下捡纸杯","chg":"由站立变蹲下"}`
  · 差：`{"at":"画面左侧靠近垃圾桶的位置","act":"一个人正在做某些动作"}`
- `chg` 是**时序信息的关键**：对比这几帧，说明动作怎么变了；
  没有变化就省略这个键，不要写「无」。
- 画面里没人时，`people` 填空数组 `[]`，不要编造。
- 人很多看不清时，只描述最清楚的 1-3 个，其余不必强行编。
"""

    # with_alert=True 时追加一段"由你决定要不要告警"的指令。
    # 这不是简单的"把 violation 复制到 alert" —— 而是明确要求模型区分
    # 「发生了」和「值得打扰人」两件事：
    #   远处模糊的一团烟、持续 1 帧就消失的动作，可以 violation=true 但 alert=false。
    # 这正是让 AI 判断的价值所在，也是最容易被实现成"两个字段永远相等"的地方。
    alert_block = ""
    if with_alert:
        alert_block = """
# 告警判定（本次任务的重要部分）
每个行为除了判断"有没有发生"，还要判断**此刻是否已经值得告警**：
- `alert` 字段填 true / false，`alert_reason` 用一句话说明理由。
- **violation 与 alert 不是同一件事**，请分开判断：
  · `violation=true, alert=false` —— 确实发生了，但画面证据还不足以打扰别人。
    例如：只在某一帧边缘看到疑似烟雾、人物远小到看不清动作、行为已结束且人已离开。
  · `violation=false, alert=true` —— 不应出现这种组合。
  · `violation=true, alert=true` —— 行为清楚、动作链条完整，现在就该提醒。
- 判 alert=true 的门槛：**你愿意仅凭当前这几帧就去提醒现场的人**吗？
  愿意才填 true。犹豫就填 false，并说明还缺什么证据。
- 不要为了"多报点"而放宽：误报会让人关掉系统，代价高于漏报。
"""

    # 长度纪律里那两处提到 alert 的地方必须按开关生成 ——
    # 关掉告警判定时还提 alert_reason，模型就会凭空多输出一个没人要的字段
    # （实测被测试用例抓到："非 ai 模式不该出现 alert_reason"）。
    alert_key_hint = " / `alert_reason`" if with_alert else ""
    alert_line = ("\n  `alert_reason` 只在 `alert=true` 时写。" if with_alert else "")

    # 「一行写完」的字面示例。
    # ⚠️ 用**纯字符串拼接**拼出来，不要用 f-string、也不要用 .format()：
    #    JSON 里的花括号和这两种语法都会打架（f-string 会把 `{"scene"...}` 当成
    #    格式占位符直接抛 ValueError；.format() 则要求把每个 { 写成 {{，
    #    一旦拼接顺序没注意就会漏掉转义，输出成 {{"at":...}} 这样的坏示例）。
    #    这个坑我两种都踩过一遍，所以这里最简单直白地拼。
    _b0 = behaviors[0].id if behaviors else "id1"
    _b1 = behaviors[1].id if len(behaviors) > 1 else "id2"
    _ex_alert = (',"alert":true,"alert_reason":"动作完整"' if with_alert else "")
    _ex_people = (',"people":[{"at":"左侧近处","act":"蹲下捡纸杯"}]'
                  if with_actions else "")
    compact_example = (
        '{"scene":"谁在哪做什么","violations":['
        '{"behavior_id":"' + _b0 + '","violation":true,"confidence":0.9,'
        '"observable":"第3帧手由持物变空，物体落地"' + _ex_alert + '},'
        '{"behavior_id":"' + _b1 + '","violation":false}'
        ']' + _ex_people + '}'
    )

    return f"""你是一个公共场所视频行为分析助手。你会看到同一时间窗口内的若干连续视频帧（按时间先后排列）。

# 任务
逐一检查下列行为是否发生，并只依据画面中**可见的证据**判断。
{alert_block}{action_block}
# 行为定义
{_render_behaviors(behaviors)}

# 判定纪律（必须遵守）
1. **先描述，再判断**：`observable` 只写画面上真实可见的事实，不写推测。
2. **证据不足即判 false**：宁可漏报，不可凭"看起来像"就报 true。监控场景里
   误报的代价（打扰、失去信任）通常高于漏报。
3. **逐条排除反例**：对每个行为，先想一遍"反例清单里有没有命中的"，命中就判 false。
4. **只报你确实看到的**：不要因为"这个场景通常会有违规"而推断存在违规。
5. **人不在画面里就不报**：没有可辨认的人时，所有需要人的行为一律 false。
6. **不要把地面上的物体当成人**：只看可辨认的人形轮廓。

# 输出长度纪律（直接影响响应速度，必须遵守）
- ⚠️ **必须输出紧凑的单行 JSON**：不要换行、不要缩进、不要多余空格。
  美化输出（每个字段一行、缩进两格）会凭空多出 30%~50% 的 token，
  而 token 就是用户的等待时间 —— 实测每 token 约 28 毫秒。
  即整个对象在一行里写完，键值之间不加空格。
- 未命中（`violation=false`）的行为：**只写 `behavior_id` 和 `violation`**，
  其余键（`confidence` / `observable`{alert_key_hint}）**全部省略**。
  但**判断过程不能省** —— 你必须先逐条排除反例再下结论，只是不必写出来。
- 命中（`violation=true`）的行为：必须写 `confidence` 与 `observable`
  （≤25 字，写支撑结论的动作链条，如"第3帧手由持物变空，物体落地"）。{alert_line}
- `scene` 一句话，**不超过 20 字**，只写"谁在哪做什么"。
- 除 JSON 外不要输出任何文字。输出越短，用户等待越少。
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


def _few_shot(bs: BehaviorSet, with_actions: bool = False) -> list[tuple[str, str]]:
    """返回 [(用户提问, 模型应答)] 示例。

    两个示例一正一负。负例用"最容易误报"的场景，教学效果最好。
    示例里的 behavior_id 动态取自实际启用的行为，避免出现不存在的 id。
    """
    behaviors = bs.enabled
    ids = [b.id for b in behaviors]

    def skeleton() -> list[dict]:
        # 未命中只写两个键（见"输出长度纪律"）—— 示例必须和规则完全一致，
        # 否则 few-shot 会教坏模型（踩过一次：示例带 observable、规则说可省略，
        # 结果模型跟着示例写长输出）
        return [{"behavior_id": i, "violation": False} for i in ids]

    # ── 负例：地上有垃圾，但没人扔 ──
    # ⚠️ 这个示例的 observable **必须留空**，因为指令就是这么要求的
    #   （"判 false 时留空 observable"，为压延迟）。
    #   踩过的坑：第一版指令说"必须填写"、示例却留空，模型跟着示例走，
    #   结果所有行为都不写依据 —— few-shot 的示范作用永远大于文字规定，
    #   所以两者必须一致。现在指令也改成留空，两边对齐了。
    neg = skeleton()
    neg_scene = "空无一人的走廊，地面散落纸屑与空瓶。"

    # ── 正例：确实在扔（命中的行为必须给出可见依据）──
    pos = skeleton()
    if "littering" in ids:
        for v in pos:
            if v["behavior_id"] == "littering":
                v.update(
                    violation=True,
                    confidence=0.86,
                    observable="第3帧手由持物变空，物体落地",
                )
    pos_scene = "走廊里一人经过垃圾桶旁。"
    if with_actions:
        # 紧凑动作示例（at/act/chg）：字段短、内容具体，示范"怎么写得又快又准"
        pos_people = [{"at": "左侧近处", "act": "把纸杯扔向垃圾桶",
                       "chg": "由站立变抛掷"}]
        neg_people: list[dict] = []
    else:
        pos_people = neg_people = None

    pre = "请分析这几帧画面，按 Schema 输出 JSON。"

    def _doc(scene: str, viol: list[dict], people: list[dict] | None) -> str:
        obj: dict = {"scene": scene, "violations": viol}
        if people is not None:
            obj["people"] = people
        # separators 去掉所有多余空格 —— 示例是模型模仿的对象，
        # 示例紧凑，模型才会紧凑（这几字节在真实输出里会放大成几十个 token）
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    return [
        (f"{pre}\n\n（示例A：负例 —— 有垃圾但无人丢弃）",
         _doc(neg_scene, neg, neg_people)),
        (f"{pre}\n\n（示例B：正例 —— 完整的丢弃动作链条）",
         _doc(pos_scene, pos, pos_people)),
    ]


def build_messages(
    bs: BehaviorSet,
    frame_images: list,
    frame_labels: list[str] | None = None,
    with_few_shot: bool = True,
    with_alert: bool = False,
    with_actions: bool = False,
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
    """
    messages: list[dict] = [
        {"role": "system",
         "content": build_system_prompt(bs, with_alert=with_alert,
                                        with_actions=with_actions)}
    ]

    if with_few_shot:
        for user_text, assistant_text in _few_shot(bs, with_actions=with_actions):
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})

    labels = frame_labels or [f"帧{i + 1}" for i in range(len(frame_images))]
    content: list[dict] = [
        {
            "type": "text",
            "text": "以下是同一时间窗口内的连续帧，按时间先后排列："
            + "、".join(labels)
            + "。请按要求输出 JSON。",
        }
    ]

    for label, img in zip(labels, frame_images):
        content.append({"type": "text", "text": f"[{label}]"})
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


def validate(obj: dict, bs: BehaviorSet, *, strict: bool = True) -> tuple[bool, list[str]]:
    """校验模型输出是否符合 Schema。返回 (是否合法, 问题列表)。

    校验失败 ⇒ 上层重试。这是让"自定义行为"能安全扩展的关键：
    用户往 YAML 里加行为后，模型偶尔会漏项或造 id，必须挡住。

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


def sanitize(obj: dict, bs: BehaviorSet) -> dict:
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
    out: dict[str, Any] = {
        "scene": str(obj.get("scene", "") or ""),
        "violations": {},
        # people：画面里每个人在做什么（仅 with_actions=True 时模型会给）。
        # 归一化成统一结构，前端不必再判断字段缺失。
        "people": [],
    }

    for p in (obj.get("people") or []):
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
        try:
            conf = float(item.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        out["violations"][b.id] = {
            "violation": bool(item.get("violation")),
            "confidence": max(0.0, min(1.0, conf)),
            "observable": str(item.get("observable", "") or ""),
            "person_desc": str(item.get("person_desc", "") or ""),
            "bbox": item.get("bbox") if isinstance(item.get("bbox"), list) else None,
            # 模型自判的告警决定（aggregate.mode: ai）。
            # alert 缺省时回落成 None（而不是 False）—— "模型没表态" 和
            # "模型明确说不用告警"必须能区分，否则上层没法决定回退策略。
            "alert": (bool(item["alert"]) if isinstance(item.get("alert"), bool) else None),
            "alert_reason": str(item.get("alert_reason", "") or ""),
            "missing": False,
        }
    return out
