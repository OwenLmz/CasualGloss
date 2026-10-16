# Baseline rationale for the ICASSP 2027 study

## Minimal experiment set

1. **Offline Fine-tune 1**: full English sentence to pseudo gloss. This is the
   non-streaming quality reference, not a latency competitor.
2. **Fixed Wait-k** with `k in {1, 3, 5, 7}`: the standard prefix-to-prefix
   simultaneous translation baseline. Reporting the full curve avoids choosing
   a favorable `k` on the locked test set.
3. **v1 lexical policy**: the original project policy.
4. **v2.1 semantic causal alignment**: an ablation showing the effect of the
   semantic eligibility rule before the state-machine correction.
5. **v2.2 proposed policy**: semantic eligibility plus corrected
   `WRITE-keeps-prefix` state transitions.

All streaming systems use the same Qwen2.5-7B base checkpoint, Fine-tune 1
initialization, LoRA budget, pseudo-gloss targets, parent splits, decoding, and
locked 500-parent evaluation. The only intended difference is the streaming
policy supervision.

## Literature basis

- Ma et al., **STACL: Simultaneous Translation with Implicit Anticipation and
  Controllable Latency**, ACL 2019, introduced the prefix-to-prefix Wait-k
  framework and Average Lagging. This is the primary methodological baseline:
  https://aclanthology.org/P19-1289/
- Arivazhagan et al., **Monotonic Infinite Lookback Attention for Simultaneous
  Machine Translation**, ACL 2019, learned an adaptive monotonic schedule and
  compared against Wait-k. It supports the fixed-versus-adaptive framing, but
  reproducing MILk would require a different attention architecture and is not
  necessary for the first ICASSP submission:
  https://aclanthology.org/P19-1126/
- Zheng et al., **Simultaneous Translation Policies: From Fixed to Adaptive**,
  ACL 2020, showed that fixed policies remain strong and useful components of
  adaptive policies. It motivates reporting multiple fixed latency points:
  https://aclanthology.org/2020.acl-main.254/
- Yin et al., **SimulSLT: End-to-End Simultaneous Sign Language Translation**,
  ACM Multimedia 2022, used Wait-k for simultaneous sign-video-to-text
  translation. Its direction differs from this project, but it demonstrates
  that Wait-k is an established baseline in simultaneous sign-language work:
  https://arxiv.org/abs/2112.04228

## Deliberately deferred comparisons

MILk, reinforcement-learning policies, and end-to-end sign-video models are
scientifically relevant but would introduce new architectures, modalities, and
training budgets. Before the September 16 deadline, they are deferred unless
the proposed method fails to produce a defensible quality-latency comparison
against Wait-k. This keeps the experiment matrix focused on the paper's causal
alignment claim.
