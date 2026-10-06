"""One loader for the eval scripts: a laya checkpoint (HF id or local dir) or a Decision 2.0 package (`vllm-sr/...`, the
decision models of vLLM Semantic Router) or a Clef release (`Cloudflare/...`); both take the same state + questions and
return the same answer fields.

    agent = load_agent("vllm-sr/Decision-2.0-Kai-0.6B")
    agent.predict(text, questions)["answers"]
"""
import json
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


class ClefAgent:
    """`predict` of laya.Agent over a Clef release (Cloudflare/clef-flash): the repo's own joint_schema_model.py loads the
    backbone through transformers, so a 4-bit quantisation config can be passed to fit the 9B on a 16 GB card. Clef cuts an
    over-long state itself (max_length 16,384)."""

    def __init__(self, repo, four_bit=True):
        import torch
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo)
        sys.path.insert(0, path)
        from joint_schema_model import ClefModel, JointSchemaHead, systemone
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        kw = {}
        if four_bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        # load_release_model of the repo, minus AutoProcessor: the image / video processor wants torchvision and text-only
        # records never touch it (encode_record only needs the tokenizer).
        backbone = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map={"": "cuda"}, **kw)
        backbone.config.use_cache = False
        head = JointSchemaHead(**json.loads(open(os.path.join(path, "joint_head_config.json"), encoding="utf-8").read()))
        head.load_state_dict(load_file(os.path.join(path, "joint_head.safetensors")), strict=True)
        self.model = ClefModel(backbone, head.to(device="cuda", dtype=torch.bfloat16)).eval()
        self.processor = type("TokenizerOnly", (), {"tokenizer": AutoTokenizer.from_pretrained(path)})()
        self._systemone = systemone
        self.truncated = 0

    def predict(self, state, questions):
        return self._systemone(self.model, self.processor, {"model": "clef", "state": state, "questions": questions})


def load_agent(path, device="cuda"):
    if path.startswith("vllm-sr/"):
        return Decision2Agent(path)
    if path.startswith("Cloudflare/"):
        return ClefAgent(path)
    import laya
    return laya.Agent(path, device=device)
