"""LLM interface using Ollama for local inference."""

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Generator
import ollama

from config import LLMConfig


class LLMError(Exception):
    """Base LLM error."""


class LLMTimeoutError(LLMError):
    """LLM request exceeded timeout."""


class LLMConnectionError(LLMError):
    """Cannot reach Ollama server."""


class LocalLLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = ollama.Client(host=config.base_url)
        self.conversation_history: list[dict] = []
        self.model = self._resolve_model()

    def _resolve_model(self) -> str:
        try:
            available = {m.model for m in self.client.list().models}
            if self.config.model in available:
                return self.config.model
            if self.config.fallback_model in available:
                print(f"[LLM] {self.config.model} niedostępny — używam {self.config.fallback_model}")
                return self.config.fallback_model
        except Exception as e:
            print(f"[LLM] Nie można sprawdzić modeli: {e}")
        return self.config.model

    def _ollama_options(self, max_tokens: int | None = None) -> dict:
        return {
            "temperature": self.config.temperature,
            "num_predict": max_tokens or self.config.max_tokens,
            "num_ctx": self.config.num_ctx,
        }

    def set_system_prompt(self, prompt: str):
        self.conversation_history = [{"role": "system", "content": prompt}]

    def check_connection(self) -> None:
        """Raise LLMConnectionError if Ollama is unreachable."""
        try:
            self.client.list()
        except Exception as e:
            raise LLMConnectionError(
                f"Nie można połączyć z Ollama ({self.config.base_url}): {e}"
            ) from e

    def _chat_impl(self, user_message: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_message})
        response = self.client.chat(
            model=self.model,
            messages=self.conversation_history,
            options=self._ollama_options(),
        )
        assistant_msg = response["message"]["content"]
        self.conversation_history.append({"role": "assistant", "content": assistant_msg})
        return assistant_msg

    def chat(self, user_message: str, timeout: float | None = None) -> str:
        """Send message and get full response."""
        timeout = timeout if timeout is not None else self.config.timeout_seconds
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._chat_impl, user_message)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError as e:
                if (
                    self.conversation_history
                    and self.conversation_history[-1].get("role") == "user"
                    and self.conversation_history[-1].get("content") == user_message
                ):
                    self.conversation_history.pop()
                raise LLMTimeoutError(
                    f"Ollama nie odpowiedziała w ciągu {timeout:.0f}s "
                    f"(model: {self.model})"
                ) from e
            except Exception as e:
                if (
                    self.conversation_history
                    and self.conversation_history[-1].get("role") == "user"
                    and self.conversation_history[-1].get("content") == user_message
                ):
                    self.conversation_history.pop()
                if "connection" in str(e).lower() or "refused" in str(e).lower():
                    raise LLMConnectionError(str(e)) from e
                raise

    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        """Send message and stream response token by token."""
        self.conversation_history.append({"role": "user", "content": user_message})

        full_response = ""
        stream = self.client.chat(
            model=self.model,
            messages=self.conversation_history,
            stream=True,
            options=self._ollama_options(),
        )

        for chunk in stream:
            token = chunk["message"]["content"]
            full_response += token
            yield token

        self.conversation_history.append({"role": "assistant", "content": full_response})

    def analyze_emotion(self, text: str) -> dict:
        """Analyze emotion, estimated age group, and gender from conversation context."""
        analysis_prompt = f"""Przeanalizuj poniższą wypowiedź pacjenta dzwoniącego do placówki medycznej.
Zwróć JSON z polami:
- "emotion": jedna z [spokój, lekki_stres, stres, złość, smutek, strach, panika]
- "emotion_intensity": liczba 1-10
- "estimated_age_group": jedna z [dziecko, nastolatek, młody_dorosły, dorosły, senior]
- "estimated_gender": jedna z [kobieta, mężczyzna, nieokreślone]
- "urgency": liczba 1-10 (jak pilna jest sprawa medyczna)
- "needs_calming": boolean (czy pacjent potrzebuje uspokojenia)

Wypowiedź: "{text}"

Kontekst rozmowy (ostatnie wiadomości):
{self._get_recent_context()}

Odpowiedz TYLKO JSON-em, bez dodatkowego tekstu."""

        timeout = min(20.0, self.config.timeout_seconds)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.client.chat,
                model=self.model,
                messages=[{"role": "user", "content": analysis_prompt}],
                options=self._ollama_options(max_tokens=120),
            )
            try:
                response = future.result(timeout=timeout)
            except FuturesTimeoutError:
                print(f"[LLM] analyze_emotion timeout ({timeout:.0f}s) — używam domyślnych wartości")
                return self._default_emotion()

        try:
            content = response["message"]["content"]
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except (json.JSONDecodeError, KeyError):
            pass

        return self._default_emotion()

    @staticmethod
    def _default_emotion() -> dict:
        return {
            "emotion": "nieokreślone",
            "emotion_intensity": 5,
            "estimated_age_group": "dorosły",
            "estimated_gender": "nieokreślone",
            "urgency": 5,
            "needs_calming": False,
        }

    def _get_recent_context(self) -> str:
        recent = self.conversation_history[-6:]
        return "\n".join(
            f"{m['role']}: {m['content'][:100]}" for m in recent if m["role"] != "system"
        )
