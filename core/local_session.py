from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from core.local_ai import LocalAI, to_openai_tools


@dataclass
class _FunctionCall:
    id: str
    name: str
    args: dict


class LocalSession:
    """Gemini-Live-shaped adapter over llama-server.

    It deliberately exposes only the tiny surface Mark-LIV's existing receive
    loop needs: send_client_content(), receive(), and send_tool_response().
    """

    def __init__(self, system_prompt: str, declarations: list[dict]):
        self.client = LocalAI()
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self.tools = to_openai_tools(declarations)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._pending_tool_calls: list[dict] = []
        self.closed = False

    @staticmethod
    def _content(turns: Any) -> str | list:
        if isinstance(turns, str):
            return turns
        parts = (turns or {}).get("parts", []) if isinstance(turns, dict) else []
        text = []
        content = []
        for part in parts:
            if isinstance(part, str):
                text.append(part)
                continue
            if "text" in part:
                text.append(str(part["text"]))
            elif "inline_data" in part:
                d = part["inline_data"]
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": "data:%s;base64,%s" % (
                            d.get("mime_type", "image/png"),
                            d.get("data", ""),
                        )
                    },
                })
        if content:
            if text:
                content.append({"type": "text", "text": "\n".join(text)})
            return content
        return "\n".join(text)

    async def send_client_content(self, turns=None, turn_complete=True):
        if self.closed:
            return
        role = turns.get("role", "user") if isinstance(turns, dict) else "user"
        content = self._content(turns)
        if not content:
            return
        self.messages.append({"role": role, "content": content})
        if turn_complete:
            await self._start_generation()

    async def _start_generation(self):
        if self._worker and not self._worker.done():
            return
        self._worker = asyncio.create_task(self._generate())

    async def _generate(self):
        try:
            result = await asyncio.to_thread(
                self.client.chat, self.messages, self.tools
            )
            choice = (result.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            tool_calls = msg.get("tool_calls") or []

            if tool_calls:
                self._pending_tool_calls = tool_calls
                calls = []
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    raw = fn.get("arguments", "{}")
                    if isinstance(raw, str):
                        try:
                            args = json.loads(raw) if raw.strip() else {}
                        except json.JSONDecodeError:
                            args = {}
                    else:
                        args = raw if isinstance(raw, dict) else {}
                    calls.append(_FunctionCall(
                        id=str(tc.get("id") or ""),
                        name=str(fn.get("name") or ""),
                        args=args,
                    ))
                # Save the assistant tool-call message before the tool results.
                self.messages.append({
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": tool_calls,
                })
                await self._queue.put(SimpleNamespace(
                    data=None,
                    server_content=None,
                    tool_call=SimpleNamespace(function_calls=calls),
                ))
                return

            text = str(msg.get("content") or "").strip()
            self.messages.append({"role": "assistant", "content": text})
            sc = SimpleNamespace(
                output_transcription=SimpleNamespace(text=text),
                input_transcription=None,
                turn_complete=True,
            )
            await self._queue.put(SimpleNamespace(
                data=None,
                server_content=sc,
                tool_call=None,
            ))
        except Exception as e:
            text = f"Local AI error: {e}"
            sc = SimpleNamespace(
                output_transcription=SimpleNamespace(text=text),
                input_transcription=None,
                turn_complete=True,
            )
            await self._queue.put(SimpleNamespace(
                data=None, server_content=sc, tool_call=None
            ))

    async def send_tool_response(self, function_responses=None):
        for fr in function_responses or []:
            fid = getattr(fr, "id", "")
            name = getattr(fr, "name", "")
            response = getattr(fr, "response", {}) or {}
            self.messages.append({
                "role": "tool",
                "tool_call_id": fid,
                "name": name,
                "content": json.dumps(response, ensure_ascii=False),
            })
        self._pending_tool_calls = []
        await self._start_generation()

    async def close(self):
        self.closed = True
        if self._worker and not self._worker.done():
            self._worker.cancel()

    def send_realtime_input(self, *args, **kwargs):
        raise RuntimeError("O modo de voz local usa STT/TTS local; áudio realtime Gemini não é usado.")
