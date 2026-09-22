"""Single-GPU fine-tune of a laya checkpoint (adapted from laya's 2xT4 DDP notebook, no DDP).

    python train_single.py --items /work/thai/data/train_items.pt --out /work/thai/out/laya-th --epochs 3

Loss = REINFORCE over noisy logit samples with a proper-scoring-rule reward (laya's RLCD recipe)
plus soft cross-entropy on the target. bf16 autocast (A2 is Ampere), gradient checkpointing.
"""
import argparse
import json
import os
import random
import time

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import build_model, proper_reward


def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "target": target,
            "qtype": torch.tensor([it["qtype"] for it in items]), "label": torch.tensor([it["label"] for it in items])}


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def save_checkpoint(model, tok, cfg, path, extra):
    os.makedirs(path, exist_ok=True)
    sd = {k: v.to(torch.bfloat16).contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(path, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(path, "encoder"))
    tok.save_pretrained(os.path.join(path, "tokenizer"))
    c = dict(cfg)
    c.update(extra)
    json.dump(c, open(os.path.join(path, "rl_agent_config.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convaiinnovations/laya-multilingual")
    ap.add_argument("--items", default="/work/thai/data/train_items.pt")
    ap.add_argument("--out", default="/work/thai/out/laya-th")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--sigma-start", type=float, default=0.4)
    ap.add_argument("--sigma-end", type=float, default=0.1)
    ap.add_argument("--max-items", type=int, default=0, help="cap for smoke runs")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda")
    model_dir = snapshot_download(args.model, allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
    _fix_tokenizer_config(model_dir)
    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    try:
        model.encoder.config.reference_compile = False
    except Exception:
        pass
    model.to(device).train()

    items = torch.load(args.items, weights_only=True)  # plain lists/dicts/ints written by prep_thai.py
    if args.max_items:
        items = items[: args.max_items]
    # length-bucketed micro-batches: sort by length inside chunks of 50 micro-batches, so padding stays small
    # but the order still varies per epoch
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": enc_params, "lr": args.lr_encoder}, {"params": head_params, "lr": args.lr_head}],
                                  weight_decay=0.01)
    total_updates = max(1, (len(items) // (args.micro_batch * args.grad_accum)) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)
    print(f"train: {len(items)} items, {args.epochs} epochs, {total_updates} updates, micro {args.micro_batch} x accum {args.grad_accum}", flush=True)
    t0 = time.time()

    for epoch in range(args.epochs):
        rng = random.Random(42 + epoch)
        rng.shuffle(items)
        chunk = args.micro_batch * 50
        ordered = []
        for c in range(0, len(items), chunk):
            ordered.extend(sorted(items[c : c + chunk], key=lambda it: len(it["ids"])))
        batches = [ordered[b : b + args.micro_batch] for b in range(0, len(ordered), args.micro_batch)]
        rng.shuffle(batches)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * (epoch / max(1, args.epochs - 1))
        epoch_loss = n = 0
        optimizer.zero_grad(set_to_none=True)
        for bi, chunk_items in enumerate(batches):
            batch = collate_train_batch(chunk_items, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, act = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                    batch["marker_pos"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device))
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            eps = torch.randn((args.group,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / args.grad_accum + 0.0 * act.sum()
            loss.backward()
            if (bi + 1) % args.grad_accum == 0 or bi + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item() * args.grad_accum
            n += 1
            if n % args.log_every == 0:
                el = time.time() - t0
                print(f"  epoch {epoch + 1}/{args.epochs} step {n}/{len(batches)} loss {loss.item() * args.grad_accum:.4f} "
                      f"ce {loss_ce.item():.4f} reward {r.mean().item():.3f} lr {scheduler.get_last_lr()[0]:.2e} "
                      f"{el / 60:.1f} min, {el / (epoch * len(batches) + n):.2f} s/step", flush=True)
        print(f"=== epoch {epoch + 1} done, avg loss {epoch_loss / max(1, n):.4f}, {(time.time() - t0) / 60:.1f} min ===", flush=True)
        save_checkpoint(model, tok, cfg, os.path.join(args.out, "checkpoint_latest"), {"epoch": epoch + 1})

    # temperature calibration on a slice of the training items (as in the notebook)
    print("calibrating temperatures", flush=True)
    del optimizer, scheduler
    torch.cuda.empty_cache()
    model.eval()
    calib = items[::15][:400]
    preds = []
    with torch.no_grad():
        for c in range(0, len(calib), 16):
            cb = collate_train_batch(calib[c : c + 16], tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l_sub, _ = model(cb["input_ids"].to(device), cb["attention_mask"].to(device), cb["marker_pos"].to(device),
                                 cb["marker_mask"].to(device), cb["qtype"].to(device))
            l_np = l_sub.float().cpu().numpy()
            for i, it in enumerate(calib[c : c + 16]):
                preds.append((it["qtype"], l_np[i, : len(it["markers"])], it["target"]))
    temps = [1.0, 1.0, 1.0]
    for qt in range(3):
        sel = [(z, t) for q_type, z, t in preds if q_type == qt]
        if sel:
            temps[qt] = fit_one_temp(sel)
    print("temperatures (choice, score, noul):", [round(t, 3) for t in temps], flush=True)
    save_checkpoint(model, tok, cfg, args.out, {"fine_tuned": True, "model_name": "laya-multilingual-th", "temperature": temps,
                                               "temperature_by_options": {}, "amp_dtype": "bf16",
                                               "training": {"items": len(items), "epochs": args.epochs, "updates": total_updates,
                                                            "minutes": round((time.time() - t0) / 60, 1),
                                                            "fine_tuned_from": args.model}})
    print(f"saved to {args.out} after {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
