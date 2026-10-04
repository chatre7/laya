"""One loader for the eval scripts: a laya checkpoint (HF id or local dir) or a Decision 2.0 package (`vllm-sr/...`, the
decision models of vLLM Semantic Router, which take the same state + questions and return the same answer fields).

    agent = load_agent("vllm-sr/Decision-2.0-Kai-0.6B")
    agent.predict(text, questions)["answers"]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


class Decision2Agent:
    """`predict` of laya.Agent over a Decision 2.0 package. Decision 2.0 refuses an over-long input instead of cutting it, so
    a text that does not fit is halved until it does (counted in `truncated`)."""

    def __init__(self, repo):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(repo, trust_remote_code=True)
        self.truncated = 0

    def predict(self, state, questions):
        for _ in range(6):
            try:
                return self.model.system_one(state=state, questions=questions)
            except Exception:  # noqa: BLE001
                if not isinstance(state, str) or len(state) < 400:
                    raise
                state = state[: len(state) // 2]
                self.truncated += 1
        return self.model.system_one(state=state, questions=questions)


def load_agent(path, device="cuda"):
    if path.startswith("vllm-sr/"):
        return Decision2Agent(path)
    import laya
    return laya.Agent(path, device=device)
