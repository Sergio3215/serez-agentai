# serez-agentai

GPT-style transformer stack and agentic framework for [Serez-Code](https://serezcode.org),
written in pure `.sz` on top of [serez-ai](https://serezcode.org/docs/serez-ai): tokenizer,
causal attention, KV-cache, sampling strategies, tool calling, episodic memory and an agent loop.

```serez
import "serez-agentai"

// Character-level tokenizer ([PAD]=0, [BOS]=1, [EOS]=2, [UNK]=3)
let tok = new CharTokenizer()
tok.buildVocab("hello world")

// Decoder-only transformer: vocab, d_model, n_heads, d_ff, n_layers, max_seq
let model = new GPTModel(tok.vocab_size, 128, 4, 512, 6, 256)

// Train a step (autodiff from the core)
Autodiff.tape()
let logits = model.forward(tok.encode("hello"))
let loss = Autodiff.crossEntropyLoss(logits, target_ids, seq, tok.vocab_size)
Autodiff.backward(loss)
model.update(0.001)

// Generate — strategies: "greedy" | "temp" | "topk" | "topp"
let response = generate(model, tok, "hello", 50, "greedy", 1.0, 40, 0.9)
```

## Install

```powershell
sz install serez-agentai
```

It depends on `serez-ai` (Dense, LayerNorm, … base layers) — both install automatically.

## What's inside

- **Tokenizer** — `CharTokenizer` (char-level): `buildVocab` / `encode` / `decode`.
- **Transformer** — `GPTModel` (decoder-only, causal masking, pre-norm), `GPTBlock`,
  `CausalMHA`, `sinusoidalPE(seq, d_model)`.
- **Inference** — `KVCache` (keys/values reused across generation steps), sampling with
  `greedy`, `sampleTemp`, `topK`, `topP`, and the full `generate(...)` loop.
- **Training** — `AgentDataLoader`, `WarmupCosineScheduler` / `WarmupLinearScheduler`,
  `clipGradients`, and extra losses (`klDivLoss`, `focalLoss`, `contrastiveLoss`).
- **Agentic framework** — `Tool` / `ToolRegistry` (tool calling — the model emits
  `[TOOL:name|k=v]` and the registry executes it), `EpisodicMemory` (keyword-searchable episode
  store), and `Agent`: a perception → reasoning → action → observation loop.

## Documentation

- **[serez-agentai reference](https://serezcode.org/docs/serez-agentai)** — tokenizer, model,
  KV-cache, sampling, schedulers, losses and the agent framework, with examples, on the
  Serez-Code site.
- **[Build a GPT agent](https://serezcode.org/guides/gpt-agent)** — step-by-step tutorial:
  train a small GPT and wire it into an agent with tools and memory.
