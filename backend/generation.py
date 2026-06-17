"""Local answer generation with SmolLM2-360M-Instruct.

The model is small enough to run on CPU; we use the GPU when one is
available but never require it. No hosted API is involved anywhere.
"""

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("backend.generation")

GENERATION_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
MAX_NEW_TOKENS = 150

# SmolLM2 is a small model; without an explicit grounding instruction it
# happily invents facts, so the system prompt spells the rule out.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "information in the provided context. If the context does not contain "
    "the answer, say that you do not know. Keep the answer short and factual."
)


class Generator:
    """Holds the tokenizer and model, loaded once at startup."""

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        logger.info(
            "Loaded generation model %s on %s", GENERATION_MODEL_NAME, self.device
        )

    def answer(self, query: str, chunks: list[dict]) -> str:
        """Generate an answer for the query grounded in the given corpus entries."""
        import time

        # Cap each chunk's contribution to the prompt. Retrieval already picked the
        # right chunk; the model does not need the full 1500 chars to answer, and
        # prefill time scales with context length on a small local model.
        PROMPT_CHUNK_CHARS = 600
        context = "\n\n".join(
            f"[{chunk['title']}]\n{chunk['text'][:PROMPT_CHUNK_CHARS]}" for chunk in chunks
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]
        # return_dict=True gives input_ids plus attention_mask, which recent
        # transformers versions expect to be passed to generate together.
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)

        n_in = inputs["input_ids"].shape[1]
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,              # explicit: reuse KV cache across tokens
                pad_token_id=self.tokenizer.eos_token_id,
            )
        dt = time.perf_counter() - t0

        # Decode only the newly generated tokens, not the prompt.
        generated = output_ids[0][n_in:]
        n_out = generated.shape[0]
        logger.info("generation: %d tokens in, %d tokens out, %.1fs", n_in, n_out, dt)

        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
