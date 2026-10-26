# Nanbeige4.3-33B on one M5 Ultra (TensorFold native, the bf16 release)

Serves [Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B), Nanbeige's 33B dense Llama-architecture
model with a 512K-token context, from its bf16 release as published, with TensorFold's native Metal engine, as an
OpenAI-compatible server with reasoning, tool calls, structured output, the prompt cache and two streams. Nothing is
converted: the weights are 61.7 GiB, which a 128 GB or 256 GB Mac holds as released, so there was no reason for a
lossy quantization.

The engine's `llama` Metal family is new with this recipe: branch `nanbeige43` on
[BobClawblaw/TensorFold](https://github.com/BobClawblaw/TensorFold/tree/nanbeige43), pinned in `recipe.toml`.

## Quick start

Needs an M5 Mac with the weights' 62 GiB plus a stream's caches free (the M5's tensor units; M1-M4 are untested),
Python 3.11+, git, `hf` and [Zig 0.17.0](https://ziglang.org/download/).

```bash
git clone https://github.com/BobClawblaw/TensorFold ~/github/TensorFold
hf download Nanbeige/Nanbeige4.3-33B --revision c1093c501cc6460427c0ac634db7223d0450c0ee --local-dir ~/models/Nanbeige4.3-33B
cp local.example.toml local.toml      # then set zig (and paths, if yours differ)
python3 -m mac build                  # the pinned engine and libtfgrammar, under ~/.cache/nanbeige43-33b
python3 -m mac up                     # http://127.0.0.1:8896/v1, model "nanbeige"
python3 -m mac chat "Write a Python function that merges overlapping intervals."
```

`python3 -m mac down|status|logs|command|validate` stop, inspect and check it. To serve the LAN on login:
`python3 -m mac launchd --port 8888` writes `~/Library/LaunchAgents/com.nanbeige.nanbeige43-33b.plist` (not loaded),
then `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.nanbeige.nanbeige43-33b.plist`.

## Settings

`recipe.toml` holds the defaults; `local.toml` overrides them for one Mac, and an environment variable of the same
name in upper case overrides both (`PORT=8888 HOST=0.0.0.0 python3 -m mac up`).

| setting | default | |
|---|---|---|
| `context` | 131072 | a stream's window; the release allows 524,288 (`context = 524288`, `parallel = 1`, `load_limit_gb = 192`) |
| `parallel` | 2 | streams decoded together, each reply equal to its run alone |
| `max_tokens` | 65536 | a reply's default ceiling; thinking is on by default and counts |
| `temperature`, `top_p`, `top_k` | 1.0, 0.95, off | the model card's agentic settings for requests that set none (it recommends 0.6 for reasoning tasks) |
| `prompt_cache_gib` | 8 | kept prompt states between turns; they give way to streams |
| `load_limit_gb` | 150 | weights, scratch and every stream's caches together (`LLAMA_LOAD_LIMIT_GB`; 0: 75% of RAM) |
| `drafts` | true | the lane engine's copy drafts (replies are identical with them off) |

**Memory.** The weights take 61.7 GiB. Every one of the 52 layers keeps bf16 keys and values, 208 KiB a token: 26 GiB
a stream at 131,072, 104 GiB at 524,288. Streams and kept prompts share what `load_limit_gb` leaves; a stream that
needs more room evicts kept prompts first. 150 GB holds two 131,072-token streams; one 524,288-token stream needs the
75% default of a 256 GB Mac (`load_limit_gb = 192`).

**Thinking** is on by default, as in the chat template: replies reason in `reasoning_content` before the answer in
`content`. A request sends `"chat_template_kwargs": {"enable_thinking": false}` to turn it off. The server forwards
only the thinking switch to the template: earlier turns' reasoning is left out of the prompt (the template's
`preserve_thinking` default) and tool calls take the template's XML form (its `tool_call_format` default), which the
server parses into `tool_calls`.

## What works, and how it was checked

All on one M5 Ultra (256 GB, macOS 27), engine commit fa5b1f98, checkpoint revision c1093c50, with an idle
97 GiB server of another model loaded beside it (the evidence's memory readings include it). The speed and prefill
evidence names the commit as it was when measured (738e1eb2): the same family code before a docs-only amend and a
rebase onto TensorFold main, whose logits, greedy replies and served reply are identical to the pinned build's; the
quick evidence is the pinned build's own.

- **Text, greedy and sampled.** Against Hugging Face's own code (transformers 4.57.1, bf16 on the CPU) on the release:
  1,500 positions teacher-forced, mean KL 0.0015, top-1 99.1%; 64 positions after a 9,000-token prompt, mean KL
  0.0013, top-1 100%. Six greedy chats of 64 tokens (code, prose, Chinese, a document, with and without thinking):
  three equal HF's token for token; the other three part at a position where the engine's two best logits are the
  same bf16 value (a tie, positions 19, 30 and 52) and are the model's own text either way. One-row steps and
  windows of 7 or 16 rows give the same logits bit for bit after prompts of 64 and 1,500 tokens. Seeded sampling
  repeats (9/9).
- **Long context.** A passphrase at 10%, 50% and 90% of a 128,550-token prompt comes back exactly on the default
  131,072-token server; a 131,132-token cold prompt runs on the 262,144-token setting (`evidence/bf16-prefill-256k`).
- **Tools and structured output.** The harness's structured check (json_object, json_schema, regex, choice,
  grammar) 18/18 and drafted == plain 12/12; its tool cases (auto, required, named, thinking, streamed) 5/5.
- **Prompt cache.** A turn resumed from a kept state (5,493 of 5,649 tokens) equals the same request after a
  restart, and starts in 0.34 s against 4.5 s cold.
- **Concurrency.** A 60-turn soak (3 chats of 20 turns beside a decoding stream) and 12 concurrent replies each equal
  the request alone; the suite's streaming 4/4, multi-turn recall 8/8, stop 3/3, thinking 10/10 and max_tokens 3/3.
- **Drafts.** The checkpoint has no draft head; the engine's copy drafts give the same replies as `drafts = false`.
- **Vision.** Not applicable: Nanbeige4.3-33B is text-only.

## Measured

Through [recipe-harness](https://github.com/BobClawblaw/recipe-harness) (target `nanbeige43-33b-m5`, the qwen Spark
recipe's ruler at suite commit 277c842a1481), greedy, 256-token replies; the raw results are in `evidence/`.

| streams | prose | code | structured | list | first token |
|---:|---:|---:|---:|---:|---:|
| 1 | 16.4 | 16.3 | 16.4 | 16.4 | 0.23-0.26 s |
| 2 | 31.8 | 28.0 | 31.7 | 31.7 | 0.34-0.38 s |

Tokens a second, all streams together (`evidence/bf16-speed`). One stream reads the 61.7 GiB of weights each step
(61 ms); two streams read them once for both rows (66 ms a round). Sampled replies decode at 16.2-16.3 tokens/s, the
same as greedy.

| cold prompt | 2,109 | 8,254 | 32,827 | 61,502 | 131,132 tokens |
|---|---:|---:|---:|---:|---:|
| tokens a second | 1,259 | 1,313 | 1,074 | 866 | 563 |
| time to first token | 1.68 s | 6.29 s | 30.6 s | 71.0 s | 233 s |

A fresh server, each prompt starting with words no earlier one did (`evidence/bf16-prefill`; the 131,132-token prompt
on the 262,144-token setting, `evidence/bf16-prefill-256k`; with thinking on, within 4%). A multi-turn chat's third
turn, resumed from 5,493 of its 5,649 tokens, starts in 0.34 s against 4.5 s cold (`evidence/bf16-quick`).

## Performance options

Each keeps the replies' bits; the numbers are this Mac's (`evidence/`).

- **Streams (`parallel`).** One stream decodes at 16.4 tokens/s, bound by reading the weights each step. Two streams
  give 28-32 tokens/s in all (66 ms a round against 61), each reply equal to its run alone; the first token moves
  from 0.23-0.26 s to 0.34-0.38 s. The engine's row kernel holds up to 32 rows a launch; the caches set the count.
- **Weights.** The release's bf16 is exact and is the only form served; no quantized conversion is offered.
- **Context and memory (`context`, `load_limit_gb`).** 208 KiB a token: 150 GB holds two 131,072-token streams, 192 GB
  one of 524,288. Cold prompts run 1.1-1.3k tokens/s to 33k tokens, 866 at 61k and 563 at 131k.
- **Prompt cache (`prompt_cache_gib`).** A resumed turn starts in 0.34 s against 4.5 s cold, with the same reply;
  kept states give way to streams that need the room.
- **Copy drafts (`drafts`).** On by default from the lane core, replies equal to `drafts = false`; they accepted 0-30%
  of their tokens on the harness's prompts and gave no measurable decode gain (drafted 16 tokens/s, undrafted 16).
- **Thinking.** On by default (the template's); it lengthens replies, not the cost a token; prompt speed with it on is
  within 4% of off.
- **Not offered.** No int8 or fp8 cache, no fast-math, no reduced-precision sums.

## Notes

- No quality suite was run on this recipe; the checks above are exactness against the release's own code, the
  harness's serving checks and its speed rulers.
- `logprobs` requests are refused by name: the release's tokenizer decodes through a `Strip` step, which the engine's
  tokenizer does not count as token-local, so the server cannot give a token's bytes on its own.
- The family refuses what it has not been checked on: tied heads, head sizes other than 128, Llama 3's scaled RoPE,
  biased projections, and quantized checkpoints.

## License

The recipe is under the [MIT License](LICENSE). Nanbeige4.3-33B is published under Apache 2.0 on
[Hugging Face](https://huggingface.co/Nanbeige/Nanbeige4.3-33B).
