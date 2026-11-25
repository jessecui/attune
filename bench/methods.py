"""Two ways to turn a round of rankings into a policy update.

Both see the same data: groups of K candidates for one prompt, plus the order the
user ranked them in. Only ranks are used, never the persona's scores.

dpo  : pairwise DPO on the ranked candidates themselves (off-policy, no new sampling).
grpo : fit a reward model on the rankings, then sample fresh completions from the
       current policy and update with group-normalised advantages (on-policy).
"""
import random

import torch
import torch.nn.functional as F


class Group:
    """One ranked round: prompt (unsteered), candidates, and their preference order."""

    def __init__(self, prompt, completions, order, ref_logp, feats, liked=None, segments=None, rewards=None,
                 bestworst=False):
        self.prompt = prompt
        self.completions = completions
        self.order = order            # indices, best first
        self.liked = liked            # indices the user picked, when feedback is "pick"
        self.bestworst = bestworst    # the user only marked a favourite and a least favourite
        # Optional: per candidate, the (prompt, completion) pieces that make up its output,
        # for policies that act in more than one step.
        self.segments = segments
        self.rewards = rewards        # explicit per-candidate rewards, e.g. best = +1, worst = -1
        self.ref_logp = ref_logp      # base-model log-probs, for DPO
        self.feats = feats            # frozen style features, for the reward model

    def pairs(self):
        """(preferred, less preferred) index pairs implied by the feedback."""
        if self.bestworst:  # best beats every other; every other beats worst; the middle is unordered
            best, worst, middle = self.order[0], self.order[-1], self.order[1:-1]
            return [(best, i) for i in middle + [worst]] + [(i, worst) for i in middle]
        if self.liked is not None:  # picked beats not picked; no order within either set
            rest = [i for i in self.order if i not in self.liked]
            return [(w, l) for w in self.liked for l in rest]
        return [(self.order[a], self.order[b])
                for a in range(len(self.order)) for b in range(a + 1, len(self.order))]


def _pick_group(groups, step, rng):
    # Alternate between the newest round and a replayed older one.
    if step % 2 == 0 or len(groups) == 1:
        return groups[-1]
    return rng.choice(groups[:-1])


class DPO:
    name = "dpo"

    def __init__(self, policy, lr=1e-4, beta=0.1, steps_per_round=8, seed=0):
        self.policy = policy
        self.beta = beta
        self.steps = steps_per_round
        self.opt = torch.optim.AdamW(policy.trainable_parameters(), lr=lr, weight_decay=0.0)
        self.rng = random.Random(seed)

    def update(self, groups, pool):
        losses = []
        for step in range(self.steps):
            g = _pick_group(groups, step, self.rng)
            prompts = [g.prompt] * len(g.completions)
            # Pass 1 (no grad): the loss and its derivative w.r.t. each log-prob.
            logp = self.policy.logprobs(prompts, g.completions).cpu().requires_grad_()
            delta = logp - g.ref_logp
            w = torch.tensor([p[0] for p in g.pairs()])
            l = torch.tensor([p[1] for p in g.pairs()])
            loss = -F.logsigmoid(self.beta * (delta[w] - delta[l])).mean()
            loss.backward()
            # Pass 2: chain those derivatives through the model one sequence at a time.
            self.opt.zero_grad()
            self.policy.accumulate_grad(prompts, g.completions, logp.grad)
            torch.nn.utils.clip_grad_norm_(self.policy.trainable_parameters(), 1.0)
            self.opt.step()
            losses.append(loss.item())
        return {"loss": sum(losses) / len(losses)}


class RewardModel:
    """Linear Bradley-Terry scorer on frozen style features."""

    def __init__(self, l2=1.0):
        self.l2 = l2
        self.w = self.mu = self.sd = None

    def fit(self, groups):
        feats = torch.cat([g.feats for g in groups])
        self.mu, self.sd = feats.mean(0), feats.std(0) + 1e-6
        diffs = []
        for g in groups:
            z = (g.feats - self.mu) / self.sd
            diffs += [z[w] - z[l] for w, l in g.pairs()]
        d = torch.stack(diffs)
        w = torch.zeros(d.shape[1], requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=100)

        def closure():
            opt.zero_grad()
            loss = F.softplus(-(d @ w)).mean() + self.l2 * (w ** 2).sum() / d.shape[1]
            loss.backward()
            return loss

        opt.step(closure)
        self.w = w.detach()
        return ((d @ self.w) > 0).float().mean().item()

    def score(self, feats):
        return ((feats - self.mu) / self.sd) @ self.w


class GRPO:
    name = "grpo"

    def __init__(self, policy, lr=1e-4, steps_per_round=8, group_size=4,
                 temperature=1.0, max_new_tokens=200, seed=0):
        self.policy = policy
        self.steps = steps_per_round
        self.group_size = group_size
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.opt = torch.optim.AdamW(policy.trainable_parameters(), lr=lr, weight_decay=0.0)
        self.rng = random.Random(seed)
        self.rm = RewardModel()

    def update(self, groups, pool):
        fit_acc = self.rm.fit(groups)
        losses, spreads = [], []
        for _ in range(self.steps):
            prompt = self.rng.choice(pool)
            prompts = [prompt] * self.group_size
            samples = self.policy.generate(prompts, max_new_tokens=self.max_new_tokens,
                                           temperature=self.temperature)
            rewards = self.rm.score(self.policy.style_features(samples))
            spreads.append(rewards.std().item())
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
            n_tokens = sum(len(s) for s in samples)
            logp = self.policy.logprobs(prompts, samples).cpu()
            loss = -(adv * logp).sum() / n_tokens
            self.opt.zero_grad()
            self.policy.accumulate_grad(prompts, samples, -adv / n_tokens)
            torch.nn.utils.clip_grad_norm_(self.policy.trainable_parameters(), 1.0)
            self.opt.step()
            losses.append(loss.item())
        return {"loss": sum(losses) / len(losses), "rm_fit_acc": fit_acc,
                "reward_spread": sum(spreads) / len(spreads)}


class GRPOStyle:
    """GRPO where the group is the candidates the user ranked and the reward is the rank.

    The candidates are on-policy: each starts with a style line the policy itself
    chose (see run.py), so a ranking reinforces both that choice and the wording.
    Several clipped steps are taken per round, as in GRPO/PPO.
    """
    name = "grpo_style"
    latent = True

    def __init__(self, policy, lr=1e-4, steps_per_round=8, clip=0.2, choice_clip=0.0, seed=0):
        self.policy = policy
        self.steps = steps_per_round
        self.clip = clip
        # If set, the style choice is clipped as one action (the probability of the whole
        # style line may move by at most this share per round) instead of token by token.
        # Token-by-token clipping lets a 70-token line's probability move by many orders
        # of magnitude in one round, which is how the policy over-commits.
        self.choice_clip = choice_clip
        self.opt = torch.optim.AdamW(policy.trainable_parameters(), lr=lr, weight_decay=0.0)

    def update(self, groups, pool):
        g = groups[-1]
        k = len(g.completions)
        reward = torch.zeros(k)
        if g.rewards is not None:
            reward = torch.tensor(g.rewards, dtype=torch.float)
        elif g.liked is not None:
            # "Pick the ones you like": reward 1 for picked, 0 for the rest.
            reward[g.liked] = 1.0
            if len(g.liked) in (0, k):
                return {"loss": 0.0}  # all or none picked says nothing about relative preference
        else:
            for place, i in enumerate(g.order):
                reward[i] = k - 1 - place
        adv = ((reward - reward.mean()) / reward.std()).tolist()
        segs = g.segments or [[(g.prompt, c)] for c in g.completions]
        old = [[self.policy.token_logprobs(p, c).detach() for p, c in cand] for cand in segs]
        n_tokens = sum(len(c) for cand in segs for _, c in cand)
        losses = []
        for _ in range(self.steps):
            self.opt.zero_grad()
            total = 0.0
            for cand, a, olds in zip(segs, adv, old):
                for part, ((p, c), o) in enumerate(zip(cand, olds)):
                    logp, clip = self.policy.token_logprobs(p, c, grad=True), self.clip
                    if self.choice_clip and g.segments and part == 0:
                        logp, o, clip = logp.sum(), o.sum(), self.choice_clip
                    ratio = torch.exp(logp - o)
                    surrogate = torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
                    loss = -surrogate.sum() / n_tokens
                    loss.backward()
                    total += loss.item()
            torch.nn.utils.clip_grad_norm_(self.policy.trainable_parameters(), 1.0)
            self.opt.step()
            losses.append(total)
        if self.policy.device == "mps":
            torch.mps.empty_cache()
        return {"loss": sum(losses) / len(losses)}


METHODS = {"dpo": DPO, "grpo": GRPO, "grpo_style": GRPOStyle}
