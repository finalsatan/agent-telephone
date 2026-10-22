"""槽位管理（T07）。

对应《级联方案实现设计》第 4 节与《开发任务拆分》T07：

- 槽位固定 5 个（0.2）：必填 `name` / `purpose` / `callback_number`，选填 `time` / `location`。
- **回拨号码不来自 LLM**：`update_phone_from_text()` 只对 ASR 原话调用
  `core.normalize.extract_phone()`，规则校验通过才写入。`update()` 里出现的
  `callback_number` 一律忽略（T02 的解析层也已经丢弃过一次，这里是第二道防线）。
- 状态流转：新值 → `unconfirmed`；改动已有值 → `corrected`；`confirm()` → `confirmed`。
- 同一槽位追问超过上限后“标记 unconfirmed 并跳过”：由 `abandon()` 记录，
  被放弃的槽位不再出现在 `missing_required()` 里，但仍会出现在槽位快照中
  （`status="unconfirmed"`、`value=None`），测试页面与摘要能看出“问过但没拿到”。

本模块不调用网络、不依赖 LLM，纯数据逻辑。
"""

from __future__ import annotations

from typing import Mapping

from core.normalize import extract_phone
from core.types import REQUIRED_SLOTS, SLOT_NAMES, SlotValue


class SlotStore:
    """5 个槽位的当前值与确认状态。

    `as_dict()` 用于交给 LLM 提示词（`core.prompts.build_messages`）与事件日志，
    返回的是副本，外部改动不会影响内部状态。
    """

    def __init__(self) -> None:
        self._slots: dict[str, SlotValue] = {name: SlotValue(None, "empty") for name in SLOT_NAMES}
        # 追问次数用尽仍拿不到的必填槽位。值为空且被放弃时 status 记为 unconfirmed。
        self._abandoned: set[str] = set()

    # ---- 读取 -------------------------------------------------------------

    def get(self, name: str) -> SlotValue:
        """按名取槽位。名字必须是 0.2 约定的 5 个之一，拼错立刻 KeyError。"""
        return self._slots[name]

    def value(self, name: str) -> str | None:
        return self._slots[name].value

    def has_value(self, name: str) -> bool:
        return bool(self._slots[name].value)

    def is_abandoned(self, name: str) -> bool:
        return name in self._abandoned

    def missing_required(self) -> list[str]:
        """还没有值、也没被放弃的必填槽位（顺序同 REQUIRED_SLOTS）。"""
        return [
            name
            for name in REQUIRED_SLOTS
            if not self.has_value(name) and not self.is_abandoned(name)
        ]

    def as_dict(self) -> dict[str, SlotValue]:
        """槽位快照（副本），键顺序固定为 SLOT_NAMES。"""
        return {name: SlotValue(slot.value, slot.status) for name, slot in self._slots.items()}

    # ---- 写入 -------------------------------------------------------------

    def update(self, slots_update: Mapping[str, object] | None) -> dict[str, SlotValue]:
        """应用 LLM 给出的槽位更新，返回真正发生变化的槽位（供 `slot_update` 事件）。

        规则（设计文档第 4 节）：
        - `callback_number` 一律忽略——号码只由 `update_phone_from_text()` 写入；
        - 白名单外的槽位名、非字符串、空白值一律忽略（不报错，避免一轮脏数据中断通话）；
        - 与当前值相同的重复值不改变状态，避免把已 `confirmed` 的槽位降回 `unconfirmed`；
        - 原本为空 → `unconfirmed`；已有值但被改写 → `corrected`。
        """
        changed: dict[str, SlotValue] = {}
        for name, value in (slots_update or {}).items():
            if name == "callback_number" or name not in self._slots:
                continue
            if not isinstance(value, str):
                continue
            slot = self._set(name, value)
            if slot is not None:
                changed[name] = slot
        return changed

    def update_phone_from_text(self, text: str, *, allow_landline: bool = False) -> bool:
        """从对方原话（ASR 最终文本）里提取回拨号码，提取到则写入并返回 True。

        `allow_landline=True` 表示调用方当前正在追问号码（T03b），此时不带上下文的
        7–8 位座机号也接受；其余情况只有 11 位手机号或带关键词的座机号才算数。
        """
        phone = extract_phone(text or "", allow_landline=allow_landline)
        if not phone:
            return False
        self._set("callback_number", phone)
        return True

    def confirm(self, name: str) -> bool:
        """把槽位标为 confirmed；空槽位无法确认，返回 False。"""
        slot = self._slots[name]
        if not slot.value:
            return False
        if slot.status == "confirmed":
            return False
        self._slots[name] = SlotValue(slot.value, "confirmed")
        return True

    def confirm_all(self) -> list[str]:
        """确认所有有值的槽位（CONFIRM 状态收到肯定答复时用），返回本次新确认的槽位名。"""
        return [name for name in SLOT_NAMES if self.confirm(name)]

    def abandon(self, name: str) -> bool:
        """追问次数用尽时放弃该槽位：标 unconfirmed（值为空）并跳过。

        已经有值的槽位不会被放弃（返回 False）。
        """
        slot = self._slots[name]
        if slot.value:
            return False
        self._slots[name] = SlotValue(None, "unconfirmed")
        self._abandoned.add(name)
        return True

    # ---- 内部 -------------------------------------------------------------

    def _set(self, name: str, value: str) -> SlotValue | None:
        """写入单个槽位，返回新的 SlotValue（无变化时返回 None）。"""
        value = value.strip()
        if not value:
            return None
        current = self._slots[name]
        if current.value == value:
            return None
        status = "corrected" if current.value else "unconfirmed"
        slot = SlotValue(value, status)
        self._slots[name] = slot
        self._abandoned.discard(name)  # 放弃过的槽位后来又被补上
        return slot
