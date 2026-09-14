"""Budgeted conversation memory and opt-in POSIX shared-filesystem snapshots."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import warnings
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field
import tiktoken

from .schema import Message


class ContextConfig(BaseModel):
    token_limit: int = Field(default=80000, ge=256)
    reserve_tokens: int = Field(default=0, ge=0)
    keep_recent_messages: int = Field(default=12, ge=1)
    strategy: Literal["summary", "recent"] = "summary"
    summary_chars: int = Field(default=4000, ge=128)
    summary_input_chars: int = Field(default=24000, ge=256)
    encoding: str | None = None
    safety_margin: float = Field(default=1.15, ge=1)
    persistence: bool = False
    storage_dir: str = ".mini_param_agent/context"
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,80}$")
    resume: bool = False
    enable_recall: bool = False


class TokenCounter:
    """Model-aware text estimate, not a provider-exact multimodal tokenizer."""

    def __init__(self, config: ContextConfig, model: str):
        self.margin = config.safety_margin
        if config.encoding:
            self.encoding = tiktoken.get_encoding(config.encoding)
        else:
            try:
                try:
                    self.encoding = tiktoken.encoding_for_model(model)
                except KeyError:
                    self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.encoding = None
                warnings.warn("Tokenizer unavailable; using conservative UTF-8 byte estimate", RuntimeWarning)
        self.calibration = 1.0

    def raw(self, messages, tools=()):
        payload = {"messages": [m.model_dump(exclude_none=True) for m in messages],
                   "tools": [t.to_openai_schema() if hasattr(t, "to_openai_schema") else
                             {"name": t.name, "description": getattr(t, "description", ""),
                              "parameters": getattr(t, "parameters", {})} for t in tools]}
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        # Special-token-looking user input must remain ordinary text.
        return (len(self.encoding.encode(text, disallowed_special=())) if self.encoding
                else len(text.encode("utf-8"))) + 3

    def count(self, messages, tools=()):
        return math.ceil(self.raw(messages, tools) * self.margin * self.calibration)

    def observe(self, prompt_tokens, messages, tools=()):
        if prompt_tokens > 0:
            self.calibration = max(self.calibration, prompt_tokens / max(1, self.raw(messages, tools)))


class ContextStore:
    """Atomic snapshots with one previous generation and optimistic concurrency.

    Shared mounts must support flock and atomic rename. No remote cluster,
    replication, encryption, or distributed consensus is provisioned here.
    """

    def __init__(self, config, workspace):
        root = Path(config.storage_dir).expanduser()
        if not root.is_absolute():
            root = Path(workspace) / root
        namespace = hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()[:20]
        self.session_id = config.session_id or uuid4().hex
        self.directory = root / namespace / self.session_id
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.directory / "context.json"
        self.backup = self.directory / "context.previous.json"
        self.revision = None

    def _read(self):
        for path in (self.path, self.backup):
            if path.exists():
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if value["version"] != 1:
                        raise ValueError("Unsupported context version")
                    for item in value["messages"] + value["archive"]:
                        Message.model_validate(item)
                    if not isinstance(value["revision"], str):
                        raise ValueError("Invalid context revision")
                    return value
                except (ValueError, KeyError, TypeError):
                    continue
        if self.path.exists() or self.backup.exists():
            raise ValueError("Both context snapshot and backup are invalid")
        return None

    def load(self):
        value = self._read()
        if value is None:
            raise FileNotFoundError(f"No saved context: {self.path}")
        self.revision = value["revision"]
        return ([Message.model_validate(m) for m in value["messages"]],
                [Message.model_validate(m) for m in value["archive"]])

    def _atomic(self, path, value):
        fd, name = tempfile.mkstemp(dir=self.directory, prefix=".context-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def save(self, messages, archive):
        import fcntl

        with (self.directory / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            previous = self._read()
            if (previous["revision"] if previous else None) != self.revision:
                raise RuntimeError("Context session changed elsewhere; use a new session_id")
            revision = uuid4().hex
            value = {"version": 1, "revision": revision,
                     "messages": [m.model_dump(exclude_none=True) for m in messages],
                     "archive": [m.model_dump(exclude_none=True) for m in archive]}
            if previous:
                self._atomic(self.backup, previous)
            self._atomic(self.path, value)
            self.revision = revision


class ConversationContext:
    def __init__(self, config, workspace, model):
        if config.reserve_tokens >= config.token_limit:
            raise ValueError("reserve_tokens must be smaller than token_limit")
        if config.resume and (not config.persistence or not config.session_id):
            raise ValueError("resume requires persistence and an explicit session_id")
        self.config = config
        self.counter = TokenCounter(config, model)
        self.store = ContextStore(config, workspace) if config.persistence else None
        self.archive: list[Message] = []

    def cut(self, messages):
        # Round boundaries keep parallel tool calls and results together, and
        # never discard the active user request. N is a minimum, not exact.
        latest_allowed = max(1, len(messages) - self.config.keep_recent_messages)
        candidates = [i for i, m in enumerate(messages) if 1 < i <= latest_allowed and m.role == "user"]
        return max(candidates, default=1)

    def excerpt(self, messages, limit):
        # Equal per-message quotas retain metadata even after large tool output.
        quota = max(16, limit // max(1, len(messages)))
        return "\n".join(json.dumps(m.model_dump(exclude_none=True), ensure_ascii=False)[:quota]
                         for m in messages)[:limit]

    def recall(self, query, limit=5):
        def terms(text):
            words = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower()))
            return words
        wanted = terms(query)
        ranked = []
        for i, message in enumerate(self.archive):
            value = json.dumps(message.model_dump(exclude_none=True), ensure_ascii=False)
            score = len(wanted & terms(value))
            if score:
                ranked.append((score, i, value[:1600]))
        return [{"archive_index": i, "excerpt": value}
                for _, i, value in sorted(ranked, reverse=True)[:limit]]
