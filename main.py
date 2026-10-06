# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import math
import os
import re
import tempfile
import time

import aiohttp

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Record
from astrbot.api.star import Context, Star, register
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.star.star_handler import EventType, star_handlers_registry

# Genie-TTS 每角色 9 种情绪（与中间件 config.json 的 emotion.rules / roles.*.emotions 对齐）。
# 情绪 = 换参考音频，零成本，中间件 /synthesize 通过 emotion 参数接收。
EMOTION_ALIASES = {
    "neutral": ("neutral", "平静", "中性", "普通", "默认"),
    "happy": ("happy", "开心", "高兴", "快乐"),
    "question": ("question", "疑问", "提问", "疑惑"),
    "tender": ("tender", "温柔", "体贴"),
    "angry": ("angry", "生气", "愤怒"),
    "sad": ("sad", "难过", "伤心", "悲伤"),
    "shy": ("shy", "害羞", "腼腆"),
    "cold": ("cold", "高冷", "冷淡", "冷漠"),
    "surprised": ("surprised", "惊讶", "震惊"),
}
_EMOTION_LOOKUP = {
    alias: key
    for key, aliases in EMOTION_ALIASES.items()
    for alias in aliases
}
# [标签] / 【标签】，内容不允许再含括号，避免嵌套误匹配
_EMOTION_BRACKET_RE = re.compile(r"(?:\[|【)([^\[\]【】]+)(?:\]|】)")

# 默认提示文案（可在 WebUI 配置中逐条覆盖）
DEFAULT_ADMIN_ONLY_MESSAGE = "抱歉，只有管理员才能使用语音功能哦~"
DEFAULT_COOLDOWN_MESSAGE = "说得太快啦，请 {remaining} 秒后再试~"
DEFAULT_LIMIT_MESSAGE = "你 {window} 秒内已用满 {limit} 句啦，请 {remaining} 秒后再试~"


def _safe_int(value, default, minimum=None, maximum=None):
    """读取整数配置；脏数据（空值 / 非数字 / 越界）一律回退默认值。"""
    try:
        number = int(float(str(value).strip()))
    except (TypeError, ValueError):
        number = int(default)
    if minimum is not None and number < minimum:
        number = int(default)
    if maximum is not None and number > maximum:
        number = int(default)
    return number


def _safe_bool(value, default=False):
    """读取布尔配置，兼容 "true"/"false" 字符串（直接 bool("false") 会得到 True）。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on", "y"):
        return True
    if text in ("false", "0", "no", "off", "n", ""):
        return False
    return bool(default)


def _safe_str(value, default=""):
    text = str(value).strip() if value is not None else ""
    return text or default


def _format_message(template, default, **fields):
    """渲染提示文案；模板占位符写错时回退默认文案，绝不抛异常。"""
    for tpl in (_safe_str(template, ""), default):
        if not tpl:
            continue
        try:
            return tpl.format(**fields)
        except (KeyError, IndexError, ValueError):
            continue
    return default


class _RateGate:
    """非管理员频控闸门：单句冷却 + 滑动窗口限额。

    - 计数维度为「用户全局」：同一 sender_id 在群聊/私聊之间共享一份计数与冷却。
    - 只有放行的调用才写入时间戳，被拒绝的请求不会把冷却无限顺延。
    - 纯内存实现，插件重载即清零；过期条目按窗口裁剪，不会无界增长。
    """

    _PRUNE_THRESHOLD = 512

    def __init__(
        self,
        cooldown_enabled=True,
        cooldown_seconds=3,
        limit_enabled=True,
        window_seconds=60,
        limit_count=10,
    ):
        self.cooldown_enabled = bool(cooldown_enabled)
        self.cooldown_seconds = max(0, int(cooldown_seconds))
        self.limit_enabled = bool(limit_enabled)
        self.window_seconds = max(1, int(window_seconds))
        self.limit_count = max(0, int(limit_count))
        self._lock = asyncio.Lock()
        self._last_ts = {}
        self._events = {}

    @property
    def cooldown_active(self):
        return self.cooldown_enabled and self.cooldown_seconds > 0

    @property
    def limit_active(self):
        return self.limit_enabled and self.limit_count > 0 and self.window_seconds > 0

    def _prune(self, now):
        if self.limit_active:
            horizon = now - self.window_seconds
            for key in list(self._events):
                stamps = [t for t in self._events[key] if t > horizon]
                if stamps:
                    self._events[key] = stamps
                else:
                    del self._events[key]
        keep = max(self.cooldown_seconds, self.window_seconds, 1) * 2
        for key in list(self._last_ts):
            if now - self._last_ts[key] > keep:
                del self._last_ts[key]

    async def check_and_record(self, key, now=None):
        """返回 (是否放行, 拒绝类型 "cooldown"/"limit"/None, 剩余秒数)。"""
        now = time.monotonic() if now is None else float(now)
        async with self._lock:
            if (
                len(self._events) > self._PRUNE_THRESHOLD
                or len(self._last_ts) > self._PRUNE_THRESHOLD
            ):
                self._prune(now)

            if self.cooldown_active:
                last = self._last_ts.get(key)
                if last is not None:
                    elapsed = now - last
                    if elapsed < self.cooldown_seconds:
                        return False, "cooldown", self.cooldown_seconds - elapsed

            if self.limit_active:
                horizon = now - self.window_seconds
                stamps = [t for t in self._events.get(key, ()) if t > horizon]
                if len(stamps) >= self.limit_count:
                    # 最早一次调用离开窗口后才会腾出名额
                    self._events[key] = stamps
                    return False, "limit", (stamps[0] + self.window_seconds) - now
                stamps.append(now)
                self._events[key] = stamps

            self._last_ts[key] = now
            return True, None, 0.0

    def reset(self, key=None):
        if key is None:
            self._last_ts.clear()
            self._events.clear()
        else:
            self._last_ts.pop(key, None)
            self._events.pop(key, None)


@register(
    "astrbot_plugin_tts_send",
    "NaE",
    "直接使用 /tts 角色说 文本（也可用 /语音、语音，支持 [情绪] 标签），调用 live-stream-chatbot 中间件（Genie-TTS）合成音频并发送到群聊。",
    "1.4.0",
)
class TTSSend(Star):
    def __init__(self, context: Context, config: "AstrBotConfig"):
        super().__init__(context)

        if isinstance(config, dict):
            self.config = config
        elif hasattr(config, "model_dump"):
            self.config = config.model_dump()
        elif hasattr(config, "dict"):
            self.config = config.dict()
        else:
            self.config = {}

        self.middleware_url = str(
            self.config.get(
                "middleware_url",
                "http://host.docker.internal:8899/synthesize",
            )
        ).rstrip("/")
        self.timeout = int(self.config.get("timeout", 300000))
        self.max_text_length = int(self.config.get("max_text_length", 120))
        self.send_error_message = bool(
            self.config.get("send_error_message", True)
        )

        # ---- 权限与频控配置 ----
        # 权限模型：默认仅管理员可用；开启 allow_non_admin 后其他用户也可用，
        # 但受下面的冷却 / 限额约束。管理员始终可用且不受限制。
        self.allow_non_admin = _safe_bool(
            self.config.get("allow_non_admin"), False
        )
        self.permission_denied_message = _safe_str(
            self.config.get("permission_denied_message"),
            DEFAULT_ADMIN_ONLY_MESSAGE,
        )
        self.cooldown_message = _safe_str(
            self.config.get("non_admin_cooldown_message"),
            DEFAULT_COOLDOWN_MESSAGE,
        )
        self.limit_message = _safe_str(
            self.config.get("non_admin_limit_message"),
            DEFAULT_LIMIT_MESSAGE,
        )
        self.rate_gate = _RateGate(
            cooldown_enabled=_safe_bool(
                self.config.get("non_admin_cooldown_enabled"), True
            ),
            cooldown_seconds=_safe_int(
                self.config.get("non_admin_cooldown_seconds"), 3, minimum=0, maximum=86400
            ),
            limit_enabled=_safe_bool(
                self.config.get("non_admin_limit_enabled"), True
            ),
            window_seconds=_safe_int(
                self.config.get("non_admin_limit_window_seconds"), 60, minimum=1, maximum=86400
            ),
            limit_count=_safe_int(
                self.config.get("non_admin_limit_count"), 10, minimum=1, maximum=100000
            ),
        )

        logger.info(
            "TTS 权限/频控: "
            f"allow_non_admin={self.allow_non_admin} "
            f"cooldown={self.rate_gate.cooldown_active}"
            f"({self.rate_gate.cooldown_seconds}s) "
            f"limit={self.rate_gate.limit_active}"
            f"({self.rate_gate.limit_count}/{self.rate_gate.window_seconds}s)"
        )

        for h in star_handlers_registry.get_handlers_by_event_type(EventType.AdapterMessageEvent):
            if "tts_send" in h.handler_full_name:
                logger.info(
                    f"TTS handler registered: {h.handler_full_name} "
                    f"priority={h.extras_configs.get('priority')} "
                    f"filters={[type(f).__name__ for f in h.event_filters]}"
                )

    @staticmethod
    def _normalize_text(text: str) -> str:
        # 兼容消息里带 @机器人 的情况（例如 @小盐 /角色说 文本）
        return re.sub(r"^@[^\s]+\s*", "", text.strip())

    @staticmethod
    def _normalize_emotion(raw: str) -> str | None:
        name = str(raw or "").strip().lower()
        # 兼容 [emo:happy] / [情绪:开心] 前缀写法
        name = re.sub(r"^(?:emo|情绪)\s*[:：]\s*", "", name)
        return _EMOTION_LOOKUP.get(name)

    def _extract_emotion_tag(self, text: str) -> tuple[str, str | None]:
        """提取文本中第一个可识别的情绪标签并从文本中移除。

        Genie 后端下情绪切换零成本，支持两种位置：
        - 跟在角色名后：樱羽艾玛[开心]说 你好
        - 独立标签：/tts [开心] 你好 / /樱羽艾玛说 【害羞】你好
        方括号内容不是已知情绪时原样保留，不影响普通文本。
        """
        m = _EMOTION_BRACKET_RE.search(text)
        while m:
            emotion = self._normalize_emotion(m.group(1))
            if emotion:
                return text[:m.start()] + text[m.end():], emotion
            m = _EMOTION_BRACKET_RE.search(text, m.end())
        return text, None

    def _get_role_map(self) -> dict:
        role_map = self.config.get("role_map", {}) or {}
        if not role_map:
            # 默认映射：与 live-stream-chatbot/config/config.json 当前角色注释保持一致
            role_map = {
                "樱羽艾玛": "default",
                "二阶堂希罗": "NTE0",
                "诗歌剧": "聪明露娜",
                "橘雪莉": "可见性实验",
            }
        return role_map

    def _is_tts_command_text(self, text: str) -> bool:
        # 无前缀命令词：/tts 樱羽艾玛说 你好 时，AstrBot 已把唤醒前缀 "/" 剥掉，
        # 插件实际收到的是 "tts 樱羽艾玛说 你好"；"/" 形式保留以兼容唤醒词关闭的场景。
        if re.match(r"^(?:/tts|tts|/语音|语音)(?:\s+|$)", text):
            return True
        # /角色说 文本
        m = re.match(r"^/([^\s/]+)说(?:\s+|$)", text)
        if m:
            return m.group(1) in self._get_role_map()
        # 无前缀「角色说 文本」：群里用 /角色说 触发时，插件收到的就是这个形态
        m = re.match(r"^([^\s#/]+)说(?:\s+|$)", text)
        if m:
            return m.group(1) in self._get_role_map()
        return False

    def _extract_role_text(self, text: str) -> str | None:
        # 去掉 /tts、tts、/语音、语音 等命令前缀
        for prefix in ("/tts", "tts", "/语音", "语音"):
            if text == prefix:
                return None
            if (
                text.startswith(prefix)
                and len(text) > len(prefix)
                and text[len(prefix)].isspace()
            ):
                text = text[len(prefix):].strip()
                break

        role = None
        # /角色说 文本
        m = re.match(r"^/([^\s/]+)说(?:\s+|$)", text)
        if m:
            role = m.group(1)

        # 普通“角色说 文本”（无前缀；/tts 樱羽艾玛说 也走这里）
        if role is None:
            m = re.match(r"^([^\s]+?)说(?:[：:]\s*|\s+)", text)
            if m:
                role = m.group(1)

        if role:
            # 把群里用的模型显示名映射成中间件 config.json 里的角色 key
            role_map = self._get_role_map()
            if role in role_map:
                return str(role_map[role])

        return role

    def _extract_command_text(self, text: str) -> str:
        # 兼容 /tts、tts、/语音、语音
        prefixes = ("/tts", "tts", "/语音", "语音")
        for prefix in prefixes:
            if text == prefix:
                return ""
            if (
                text.startswith(prefix)
                and len(text) > len(prefix)
                and text[len(prefix)].isspace()
            ):
                text = text[len(prefix):].strip()
                break

        # /角色说 文本
        m = re.match(r"^/([^\s/]+)说(?:\s+|$)", text)
        if m:
            text = text[1:].strip()

        # 去掉开头的“角色说/角色说：”，避免中间件把角色名也朗读出来
        # 如果以后中间件能正确识别角色，再把这段去掉即可恢复角色语音。
        text = re.sub(r"^[^\s]+?说(?:[：:]\s*|\s+)", "", text)

        return text

    async def _reply_and_stop(self, event: AstrMessageEvent, message: str) -> None:
        """发送提示并终止本次事件，避免命令文本漏给 LLM 当普通聊天处理。"""
        await event.send(MessageChain().message(message))
        event.stop_event()

    def _render_message(self, template, default, remaining=0):
        """统一给提示文案提供全部占位符，避免用户自定义文案因缺字段而回退。"""
        return _format_message(
            template,
            default,
            remaining=remaining,
            cooldown=self.rate_gate.cooldown_seconds,
            limit=self.rate_gate.limit_count,
            window=self.rate_gate.window_seconds,
        )

    async def _check_access(self, event: AstrMessageEvent) -> bool:
        """权限与频控检查。返回 True 表示已拦下（提示已发送、事件已终止）。"""
        # 管理员始终可用：不受冷却 / 限额限制，也不占用计数
        if event.is_admin():
            return False

        # 默认仅管理员；开启后其他用户才进入下面的冷却 / 限额
        if not self.allow_non_admin:
            await self._reply_and_stop(
                event,
                self._render_message(
                    self.permission_denied_message, DEFAULT_ADMIN_ONLY_MESSAGE
                ),
            )
            return True

        sender_id = str(event.get_sender_id() or "")

        allowed, reason, remaining = await self.rate_gate.check_and_record(sender_id)
        if allowed:
            return False

        wait = max(1, int(math.ceil(remaining)))
        if reason == "cooldown":
            text = self._render_message(
                self.cooldown_message, DEFAULT_COOLDOWN_MESSAGE, remaining=wait
            )
        else:
            text = self._render_message(
                self.limit_message, DEFAULT_LIMIT_MESSAGE, remaining=wait
            )
        logger.info(
            f"TTS 频控拦截 user={sender_id} reason={reason} remaining={remaining:.2f}s"
        )
        await self._reply_and_stop(event, text)
        return True

    async def _synthesize(
        self, text: str, role: str | None = None, emotion: str | None = None
    ) -> str:
        url = self.middleware_url
        params = {"text": text, "return": "1"}
        if role:
            params["role"] = role
        if emotion:
            # Genie 后端按情绪换参考音频；角色缺该情绪时中间件自动回退 neutral
            params["emotion"] = emotion
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        data = b""
        content_type = ""
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                content_type = resp.headers.get("Content-Type", "")
                if resp.status != 200:
                    detail = await resp.text()
                    raise RuntimeError(
                        f"中间件返回 HTTP {resp.status}: {detail[:200]}"
                    )
                data = await resp.read()

        if not data:
            raise RuntimeError("中间件返回了空音频")

        if "mpeg" in content_type or "mp3" in content_type:
            suffix = ".mp3"
        elif "ogg" in content_type:
            suffix = ".ogg"
        else:
            suffix = ".wav"

        fd, path = tempfile.mkstemp(prefix="astrbot_tts_", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        except Exception:
            try:
                os.remove(path)
            except OSError:
                pass
            raise
        return path

    @filter.event_message_type(filter.EventMessageType.ALL, priority=9999)
    async def tts_send(self, event: AstrMessageEvent):
        raw = event.get_message_str().strip()
        norm = self._normalize_text(raw)
        # 先摘掉情绪标签（兼容 樱羽艾玛[开心]说 写法），再走命令/角色解析
        norm, emotion = self._extract_emotion_tag(norm)
        if raw.startswith("/") or "说" in raw or norm.startswith("/") or "说" in norm:
            logger.info(
                f"TTS DEBUG raw={raw!r} norm={norm!r} emotion={emotion!r} "
                f"is_cmd={self._is_tts_command_text(norm)}"
            )
        # 不是 TTS 命令时直接放行，不干扰其他插件（也不消耗冷却/限额）
        if not self._is_tts_command_text(norm):
            return
        role = self._extract_role_text(norm)
        text = self._extract_command_text(norm)
        logger.info(f"TTS command triggered: text={text!r} role={role!r} emotion={emotion!r}")
        if not text:
            await self._reply_and_stop(
                event,
                "用法：/樱羽艾玛说 你好\n"
                "也可：/tts 樱羽艾玛说 你好、/语音 樱羽艾玛说 你好\n"
                "带情绪：/樱羽艾玛[开心]说 你好（也可 /tts [happy] 樱羽艾玛说 你好）",
            )
            return

        # 权限与频控：拦下时提示已发出、事件已终止
        if await self._check_access(event):
            return

        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]

        audio_path = None
        try:
            audio_path = await self._synthesize(text, role, emotion)
        except Exception as e:
            logger.error(f"TTS 合成失败: {e}")
            if self.send_error_message:
                await event.send(
                    MessageChain().message(f"TTS 合成失败：{e}")
                )
            event.stop_event()
            return

        try:
            await event.send(MessageChain([Record.fromFileSystem(audio_path)]))
        finally:
            if audio_path:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

        event.stop_event()
