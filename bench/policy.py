"""The policy: a small chat model with a LoRA adapter, plus the few ops the loop needs."""
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Policy:
    def __init__(self, model_name, device=None, lora_r=16, lora_alpha=32):
        self.device = device or pick_device()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.eos_id = self.tok.eos_token_id
        self.pad_id = self.tok.pad_token_id
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device)
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
                         target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        self.model = get_peft_model(base, cfg)
        self.model.eval()

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def snapshot(self):
        """Copy of the adapter weights, so an update can be undone."""
        return [p.detach().clone() for p in self.trainable_parameters()]

    def restore(self, snap):
        with torch.no_grad():
            for p, s in zip(self.trainable_parameters(), snap):
                p.copy_(s)

    def prompt_ids(self, system, user):
        text = self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        return self.tok(text, add_special_tokens=False).input_ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True).strip()

    def _pad(self, seqs, side):
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            sl = slice(n - len(s), n) if side == "left" else slice(0, len(s))
            ids[i, sl] = torch.tensor(s)
            mask[i, sl] = 1
        return ids.to(self.device), mask.to(self.device)

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=200, temperature=0.0, top_p=1.0, use_adapter=True):
        """Completions as token-id lists; each ends with EOS if the model finished."""
        ids, mask = self._pad(prompts, "left")
        kwargs = dict(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                      pad_token_id=self.pad_id, eos_token_id=self.eos_id,
                      repetition_penalty=1.0, top_k=None)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None)
        if use_adapter:
            out = self.model.generate(**kwargs)
        else:
            with self.model.disable_adapter():
                out = self.model.generate(**kwargs)
        completions = []
        for row in out[:, ids.shape[1]:].tolist():
            if self.eos_id in row:
                row = row[: row.index(self.eos_id) + 1]
            completions.append(row)
        return completions

    def logprobs(self, prompts, completions, use_adapter=True, grad=False):
        """Sum of token log-probs of each completion given its prompt. Shape [n]."""
        seqs = [p + c for p, c in zip(prompts, completions)]
        ids, mask = self._pad(seqs, "right")
        is_completion = torch.zeros_like(mask, dtype=torch.bool)
        for i, (p, c) in enumerate(zip(prompts, completions)):
            is_completion[i, len(p): len(p) + len(c)] = True

        def run():
            logits = self.model(input_ids=ids, attention_mask=mask).logits[:, :-1]
            sel = is_completion[:, 1:]
            nll = F.cross_entropy(logits[sel].float(), ids[:, 1:][sel], reduction="none")
            per_token = torch.zeros(sel.shape, device=self.device, dtype=nll.dtype)
            per_token[sel] = -nll
            return per_token.sum(dim=1)

        with torch.set_grad_enabled(grad):
            if use_adapter:
                return run()
            with self.model.disable_adapter():
                return run()

    def token_logprobs(self, prompt, completion, grad=False):
        """Log-prob of each completion token for one sequence. Shape [len(completion)]."""
        ids = torch.tensor([prompt + completion], device=self.device)
        with torch.set_grad_enabled(grad):
            logits = self.model(input_ids=ids).logits[0, len(prompt) - 1: -1]
            return -F.cross_entropy(logits.float(), ids[0, len(prompt):], reduction="none")

    def accumulate_grad(self, prompts, completions, coefs):
        """Add the gradient of sum_i coefs[i] * logprob_i, one sequence at a time.

        Backpropagating a whole group at once needs several times the model's own
        memory; doing it per sequence gives the same gradient at a fraction of that.
        """
        for p, c, k in zip(prompts, completions, coefs.tolist()):
            if k == 0.0:
                continue
            (k * self.logprobs([p], [c], grad=True)).sum().backward()
        if self.device == "mps":
            torch.mps.empty_cache()

    @torch.no_grad()
    def style_features(self, completions):
        """Frozen base-model embedding of each completion, for the reward model."""
        ids, mask = self._pad(completions, "right")
        with self.model.disable_adapter():
            h = self.model(input_ids=ids, attention_mask=mask,
                           output_hidden_states=True).hidden_states[-1].float()
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1)
        log_len = mask.sum(1, keepdim=True).float().log()
        return torch.cat([pooled, log_len], dim=1).cpu()
