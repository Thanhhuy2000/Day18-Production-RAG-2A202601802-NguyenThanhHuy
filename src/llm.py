from __future__ import annotations

"""
LLM Provider Layer
==================
Một chỗ duy nhất quyết định gọi LLM nào cho: answer generation (pipeline),
enrichment (M5) và RAGAS judge (M4).

Thứ tự ưu tiên: GOOGLE_API_KEY (Gemini) → OPENAI_API_KEY (OpenAI) → None (fallback
extractive, pipeline vẫn chạy end-to-end nhưng RAGAS = 0).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (OPENAI_API_KEY, GOOGLE_API_KEY, GEMINI_CHAT_MODEL, GEMINI_JUDGE_MODEL,
                    GEMINI_EMBED_MODEL, OPENAI_CHAT_MODEL, LLM_MAX_RETRIES)

_GEMINI_MODELS: dict[str, object] = {}
_OPENAI_CLIENT = None


def provider() -> str:
    """Trả về provider đang dùng: "gemini" | "openai" | "none"."""
    if GOOGLE_API_KEY:
        return "gemini"
    if OPENAI_API_KEY:
        return "openai"
    return "none"


def has_llm() -> bool:
    return provider() != "none"


# ─── Gemini ──────────────────────────────────────────────


# Gemini 3.x bật "thinking" mặc định: thinking tokens tính chung vào max_output_tokens,
# nên budget nhỏ (100-400) sẽ bị thinking ăn hết và trả về text rỗng/cụt.
# Đặt sàn đủ rộng để phần trả lời thật luôn được sinh ra.
GEMINI_MIN_OUTPUT_TOKENS = 2048


def _gemini_model(json_mode: bool, max_tokens: int, system: str):
    """Cache model theo (json_mode, max_tokens, system) — tránh khởi tạo lại mỗi call."""
    import google.generativeai as genai

    max_tokens = max(max_tokens, GEMINI_MIN_OUTPUT_TOKENS)
    key = f"{json_mode}|{max_tokens}|{system}"
    if key not in _GEMINI_MODELS:
        genai.configure(api_key=GOOGLE_API_KEY)
        generation_config = {"temperature": 0.0, "max_output_tokens": max_tokens}
        if json_mode:
            generation_config["response_mime_type"] = "application/json"
        _GEMINI_MODELS[key] = genai.GenerativeModel(
            GEMINI_CHAT_MODEL,
            system_instruction=system or None,
            generation_config=generation_config,
        )
    return _GEMINI_MODELS[key]


# ─── Public API ──────────────────────────────────────────


def chat(system: str, user: str, json_mode: bool = False, max_tokens: int = 512) -> str | None:
    """Gọi LLM 1 lượt. Trả về text, hoặc None nếu không có provider / lỗi hết retry.

    Có retry + backoff vì Gemini free tier giới hạn RPM (429 RESOURCE_EXHAUSTED).
    """
    p = provider()
    if p == "none":
        return None

    last_error = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            if p == "gemini":
                model = _gemini_model(json_mode, max_tokens, system)
                resp = model.generate_content(user)
                return (resp.text or "").strip()

            global _OPENAI_CLIENT
            if _OPENAI_CLIENT is None:
                from openai import OpenAI
                _OPENAI_CLIENT = OpenAI()
            kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            resp = _OPENAI_CLIENT.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=0.0,
                **kwargs,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_error = e
            if attempt < LLM_MAX_RETRIES - 1:
                time.sleep(2 ** attempt * 2)  # 2s, 4s, 8s...

    print(f"  ⚠️  LLM call failed sau {LLM_MAX_RETRIES} lần thử: {last_error}")
    return None


def _ragas_compat_gemini():
    """ChatGoogleGenerativeAI vá lỗi tương thích với RAGAS 0.1.x.

    RAGAS gọi `generate_prompt(..., temperature=..., n=...)`, các kwarg này được
    truyền thẳng xuống google client và gây:
        TypeError: generate_content() got an unexpected keyword argument 'temperature'
    Ở đây ta chuyển temperature vào generation_config (đúng chỗ của nó) và bỏ `n`
    (RAGAS đã tự nhân bản prompt khi cần nhiều completion).
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    class RagasCompatGemini(ChatGoogleGenerativeAI):
        @staticmethod
        def _fix_kwargs(kwargs: dict) -> dict:
            temperature = kwargs.pop("temperature", None)
            kwargs.pop("n", None)
            gen_config = dict(kwargs.pop("generation_config", None) or {})
            if temperature is not None:
                gen_config["temperature"] = float(temperature)
            gen_config.setdefault("max_output_tokens", GEMINI_MIN_OUTPUT_TOKENS)
            kwargs["generation_config"] = gen_config
            return kwargs

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return super()._generate(messages, stop=stop, run_manager=run_manager,
                                     **self._fix_kwargs(kwargs))

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager,
                                            **self._fix_kwargs(kwargs))

    return RagasCompatGemini


def ragas_backend():
    """Trả về (llm, embeddings) cho RAGAS, hoặc (None, None) để RAGAS dùng default OpenAI.

    RAGAS 0.1.x nhận trực tiếp LangChain LLM/Embeddings và tự wrap.
    Dùng embeddings của Gemini qua API → không tốn RAM cho model local.
    """
    p = provider()
    if p == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        llm = _ragas_compat_gemini()(model=GEMINI_JUDGE_MODEL, google_api_key=GOOGLE_API_KEY,
                                     temperature=0.0)
        embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBED_MODEL,
                                                  google_api_key=GOOGLE_API_KEY)
        return llm, embeddings
    return None, None


def answer_from_context(question: str, contexts: list[str]) -> str:
    """Sinh câu trả lời grounded trên context (dùng chung cho naive baseline + production)."""
    if not contexts:
        return "Không tìm thấy thông tin."

    context_str = ("\n\n---\n\n").join(contexts)
    system = ("Bạn là trợ lý tra cứu chính sách nội bộ. Trả lời CHỈ dựa trên context được cung cấp. "
              "Trả lời ngắn gọn, trực tiếp vào câu hỏi, nêu rõ con số/điều kiện nếu có. "
              "Nếu context không chứa thông tin → trả lời đúng một câu: 'Không tìm thấy.'")
    user = f"Context:\n{context_str}\n\nCâu hỏi: {question}"

    answer = chat(system, user, max_tokens=400)
    return answer if answer else contexts[0]
