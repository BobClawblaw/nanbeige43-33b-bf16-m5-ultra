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
hf download Nanbeige/Nanbeige4.1-3B --revision 9c4555c37921c0982af8fffeee1e0a6e6b953c4f --local-dir ~/models/Nanbeige4.1-3B
cp local.example.toml local.toml      # then set zig (and paths, if yours differ)
python3 -m mac build                  # the pinned engine and libtfgrammar, under ~/.cache/nanbeige43-33b
python3 -m mac convert --drafter      # Nanbeige4.1-3B to 4-bit, the drafter (about a minute; skip with drafter = false)
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
| `drafts` | true | drafted windows (replies are identical with them off) |
| `drafter` | true | Nanbeige4.1-3B proposing the next tokens, 4-bit, verified by the 33B ([drafter](#drafter-nanbeige41-3b)); false: the lane engine's copy drafts alone |
| `variant` | bf16 | `bf16`: the release, exact; `8bit`, `4bit`: its MLX affine conversions ([variants](#8-bit-and-4-bit-variants)) |

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
bf16 quick evidence names fa5b1f98, the same code before the variants' commit; the variants' evidence names c0243b7e,
that commit's code before comment-only amends; the drafter's evidence names e1f447bf with the drafter's changes in
the tree (committed as 2478811a after the runs, the binary's SHA-256 7d06edaf as measured). The pinned commit is
2478811a.

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
- **Drafts.** The checkpoint has no draft head; Nanbeige4.1-3B drafts for it ([below](#drafter-nanbeige41-3b)):
  drafted greedy tokens equal plain greedy tokens on nine prompts in the engine's own tool, and three served
  greedy replies (a function, a word problem with thinking, an explanation) equal the undrafted server's token for
  token, the first the published build's. The quick plan passes in full with the drafter (`evidence/draft-quick`).
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
- **Drafter (`drafter`).** Nanbeige4.1-3B at 4-bit proposing the next tokens, on by default: 85-114 tokens/s on
  the harness's speed plan against 16.4 undrafted, 23-30 on ordinary chats, replies identical
  ([below](#drafter-nanbeige41-3b)). With
  `drafter = false` the lane core's copy drafts alone accepted 0-30% of their tokens on the harness's prompts and gave
  no measurable decode gain (drafted 16 tokens/s, undrafted 16).
- **Thinking.** On by default (the template's); it lengthens replies, not the cost a token; prompt speed with it on is
  within 4% of off.
- **Not offered.** No int8 or fp8 cache, no fast-math, no reduced-precision sums.

## Drafter: Nanbeige4.1-3B

The 33B's decode is bound by reading 62 GiB of weights each step. Speculative decoding has a smaller model propose
the next few tokens and the 33B verify them all in one window, which costs about as much as one step; every token
the 33B keeps is one it would have drawn itself, so replies are the same. [Nanbeige4.1-3B](https://huggingface.co/Nanbeige/Nanbeige4.1-3B)
is a plain Llama on the same tokenizer (the same 166,100 ids, merges, special tokens and end tokens), so the engine's
llama family serves it as the drafter beside the 33B:

- `python3 -m mac convert --drafter` writes its 4-bit conversion (2.63 GiB; `checksums/nanbeige4.1-3b-4bit.sha256`)
  with the same script as the 33B's variants, and `drafter = true` passes it to the server
  (`--drafter ~/models/Nanbeige4.1-3B-4bit --drafter-bits 4`).
- Before each round the drafter syncs its own cache to the stream's context (an accepted draft stays, a rejected one
  is cut), then runs greedy steps chained on the GPU, 3.6 ms each. The depth follows measured rates: the stream's
  acceptance, the step time and the 33B's window time by rows. A request takes one drafter cache (64 KiB a token, grown
  in 4,096-row steps as the chat needs) that keeps its tokens between requests, so a continued chat is not fed again.
- The drafter's weights are lossy, which changes no reply: the 33B's own draws decide every token. The bf16 release of
  the drafter (`--drafter-bits 0`) lands more drafts but takes 8.5 ms a step and gives no more tokens a second.

Checked in the engine's own tool, which decodes greedily through the drafter's windows: on eight chat prompts (128
tokens, thinking on and off, English and Chinese) and after a 1,500-token prompt (256 tokens) the drafted tokens
equal the plain greedy tokens exactly, at 24-46 tokens/s against 16.2, with 1.8-4.1 tokens landing a round
(the engine doc's table has each prompt). Served, on the harness's plans (`evidence/draft-quick`, `evidence/draft-speed`),
beside an idle 93 GiB server of another model:

| | undrafted | drafted |
|---|---:|---:|
| one stream, the ruler's prose / code / structured / list (256 tokens) | 16.4 / 16.3 / 16.4 / 16.4 tok/s | 113.5 / 111.0 / 108.5 / 84.9 |
| two streams together, the same | 31.8 / 28.0 / 31.7 / 31.7 | 63.3 / 64.3 / 63.8 / 58.3 |
| one stream's rounds | 61 ms, 1 token | 47 ms, 4.9 tokens (81% of drafts land) |
| three greedy chats, whole requests (prompt and reply) | 15.7 / 16.0 / 16.1 tok/s | 30.2 / 22.9 / 26.6 |
| first token, resumed turn | 0.23-0.26 s, 0.34 s | 0.23-0.26 s, 0.34 s |

The ruler's texts repeat themselves, which the drafter continues almost whole; ordinary chats land 2-4 tokens a
round for 1.5-3x. Two streams land fewer drafts a round each (the engine trims shared rounds' copies by its own
prior), and still double the undrafted pair. Every quick-plan check passes with the drafter; the first token keeps
its time, since the drafter reads the prompt after the model's own prompt pass (about 0.1 s for 1,500 tokens).

## 8-bit and 4-bit variants

`variant = "8bit"` or `"4bit"` (in `local.toml`, or `VARIANT=8bit`) serves an MLX affine conversion of the release,
groups of 64, made from the downloaded release by the engine's own converter (`tools/zig/llama_convert.py`, numpy
only): every projection quantized, q|k|v and gate|up fused into one matrix each, the embedding and norms kept bf16.

```bash
VARIANT=8bit python3 -m mac convert    # model_dir -> model_dir_8bit (~/models/Nanbeige4.3-33B-8bit), 33.8 GiB, about 2 minutes
(cd ~/models/Nanbeige4.3-33B-8bit && shasum -a 256 -c PATH_TO_THIS_REPO/checksums/nanbeige4.3-33b-8bit.sha256)
VARIANT=8bit python3 -m mac up
```

The 4-bit one is the same with `4bit` (18.9 GiB). The shards' hashes in `checksums/` came from numpy 2.5.4.

**What they cost.** Both are lossy conversions of a release that a 256 GB Mac holds in its own precision, so neither
is the default. Against the release through HF (bf16 on the CPU) over the same 1,500 positions, and the same six
64-token greedy chats:

| | bf16 (default) | 8-bit | 4-bit |
|---|---:|---:|---:|
| mean KL over 1,500 positions | 0.0015 | 0.0023 | 0.082 |
| top-1 agreement | 99.1% | 98.9% | 91.1% |
| 64 positions after a 9,000-token prompt: KL, top-1 | 0.0013, 100% | 0.0015, 100% | 0.046, 95.3% |
| greedy chats equal to HF's over 64 tokens | 3 of 6 (ties) | 4 of 6 (parts at 19 and 55) | 0 of 6 (parts at 4 to 16) |

One-row steps and windows still give the same logits bit for bit, and every serving check passes for each
(`evidence/8bit-quick`, `evidence/4bit-quick`: structured 18/18, tools 5/5, sampling 9/9, turns, soak 60/60,
concurrent 12/12, multi-turn 8/8).

**What they gain**, on the same engine commit (`evidence/8bit-speed`, `8bit-prefill`, `4bit-speed`, `4bit-prefill`):

| | bf16 (default) | 8-bit | 4-bit |
|---|---:|---:|---:|
| decode, 1 stream | 16.4 tok/s | 28.4-28.9 tok/s | 48.1-48.8 tok/s |
| decode, 2 streams, all | 28-32 | 52.5-54.1 | 81.3-84.8 |
| first token (short prompt) | 0.23-0.26 s | 0.13-0.15 s | 0.10-0.12 s |
| cold prompt 2.1k / 8.3k / 33k / 61k | 1,259 / 1,313 / 1,074 / 866 tok/s | 1,269 / 1,254 / 1,009 / 815 | 1,344 / 1,312 / 1,044 / 836 |
| resumed turn, first token | 0.34 s | 0.26 s | 0.22 s |

Decode is faster because a step reads 33.8 or 18.9 GiB instead of 61.7; the 8-bit prompts are a little slower
because the tensor-unit matmul dequantizes on the way in. The 4-bit conversion is a real loss: a mean KL of 0.082, one
token in eleven chosen differently, and every greedy chat parting from the release within 16 tokens. It is offered for
the speed, with that cost stated; the 8-bit one keeps the release's choices 98.9% of the time for 1.75x the tokens.

**Quality.** No quality suite was run for the variants either.

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
