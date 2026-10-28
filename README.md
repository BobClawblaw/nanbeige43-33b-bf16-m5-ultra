# Nanbeige4.3-33B on one M5 Ultra (TensorFold native, the bf16 release)

Serves [Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B), Nanbeige's 33B dense Llama-architecture
model with a 512K-token context, from its bf16 release as published, with TensorFold's native Metal engine, as an
OpenAI-compatible server with reasoning, tool calls, structured output, the prompt cache and two streams, with
[Nanbeige4.1-3B](https://huggingface.co/Nanbeige/Nanbeige4.1-3B) drafting for it (replies unchanged). Nothing is
converted: the weights are 61.7 GiB, which a 128 GB or 256 GB Mac holds as released, so there was no reason for a
lossy quantization. The drafter is served from its bf16 release as published, too.

The engine's `llama` Metal family is new with this recipe and under review upstream as
[ashhart/TensorFold#1415](https://github.com/ashhart/TensorFold/pull/1415) (the family) and
[#1417](https://github.com/ashhart/TensorFold/pull/1417) (the drafter): branch `llama-drafter-pr` on
[BobClawblaw/TensorFold](https://github.com/BobClawblaw/TensorFold/tree/llama-drafter-pr), which carries both, is
pinned in `recipe.toml`.

## Quick start

Needs an M5 Mac with the weights' 62 GiB plus a stream's caches free (the M5's tensor units; M1-M4 are untested),
Python 3.11+, git, `hf` and [Zig 0.17.0](https://ziglang.org/download/).

```bash
git clone https://github.com/BobClawblaw/TensorFold ~/github/TensorFold
hf download Nanbeige/Nanbeige4.3-33B --revision c1093c501cc6460427c0ac634db7223d0450c0ee --local-dir ~/models/Nanbeige4.3-33B
hf download Nanbeige/Nanbeige4.1-3B --revision 9c4555c37921c0982af8fffeee1e0a6e6b953c4f --local-dir ~/models/Nanbeige4.1-3B
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
| `drafts` | true | drafted windows (replies are identical with them off) |
| `drafter` | true | Nanbeige4.1-3B proposing the next tokens, verified by the 33B ([drafter](#drafter-nanbeige41-3b)); false: the lane engine's copy drafts alone |

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

All on one M5 Ultra (256 GB, macOS 27), checkpoint revision c1093c50. The serving checks, speed and prompt numbers
were measured on the pinned engine commit 50fe245a on 28 Oct 2026, with no other model loaded
(`evidence/bf16-*` without the drafter, `evidence/draft-*` with it), except the 61k and 131k cold prompts and the
256k setting, measured on the family's first build (738e1eb2, `evidence/bf16-prefill-256k`). The exactness checks
against Hugging Face and the 128k passphrase check were run on the family's first build too; the pinned build's logits over the same 1,500
positions are bit-identical to that build's.

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
  restart, and starts in 0.35 s against 4.4 s cold.
- **Concurrency.** A 60-turn soak (3 chats of 20 turns beside a decoding stream) and 12 concurrent replies each equal
  the request alone; the suite's streaming 4/4, multi-turn recall 8/8, stop 3/3, thinking 10/10 and max_tokens 3/3.
- **Drafts.** The checkpoint has no draft head; Nanbeige4.1-3B drafts for it ([below](#drafter-nanbeige41-3b)):
  drafted greedy tokens equal plain greedy tokens on nine prompts in the engine's own tool, eight fresh prompts
  served two at a time return the undrafted server's token ids, and the quick plan passes in full with the drafter
  (`evidence/draft-quick`: structured 18/18 with drafted == plain 12/12, sampling 9/9 equal to undrafted, soak 60/60,
  concurrent 12/12).
- **Vision.** Not applicable: Nanbeige4.3-33B is text-only.

## Measured

Through [recipe-harness](https://github.com/BobClawblaw/recipe-harness) (targets `nanbeige43-33b-m5` and, with the
drafter, `nanbeige43-33b-bf16draft-m5`; the qwen Spark recipe's ruler at suite commit 277c842a1481), greedy,
256-token replies, engine 50fe245a; the raw results are in `evidence/`.

| | streams | prose | code | structured | list | first token |
|---|---:|---:|---:|---:|---:|---:|
| undrafted | 1 | 16.3 | 16.2 | 16.3 | 16.3 | 0.23-0.26 s |
| undrafted | 2 | 31.7 | 27.8 | 31.4 | 31.6 | 0.34-0.39 s |
| drafted (default) | 1 | 113.9 | 108.3 | 98.6 | 98.9 | 0.23-0.26 s |
| drafted (default) | 2 | 44.6 | 59.0 | 54.4 | 55.9 | 0.35-0.39 s |

Tokens a second, all streams together (`evidence/bf16-speed`, `evidence/draft-speed`). Undrafted, one stream reads
the 61.7 GiB of weights each step (61.7 ms); two streams read them once for both rows (66.8 ms a round). The ruler's
texts repeat themselves, which the drafter continues almost whole (4.9 tokens a round, 85% of drafts kept); ordinary
chats gain less ([drafter](#drafter-nanbeige41-3b)).

| cold prompt | 2,111 | 8,253 | 32,831 | 61,502 | 131,132 tokens |
|---|---:|---:|---:|---:|---:|
| tokens a second | 1,250 | 1,303 | 1,062 | 866 | 563 |
| time to first token | 1.69 s | 6.33 s | 30.9 s | 71.0 s | 233 s |

A fresh server, each prompt starting with words no earlier one did, undrafted (`evidence/bf16-prefill`; with thinking
on, within 1%; with the drafter, 2-4% slower, `evidence/draft-prefill`). The 61k and 131k prompts were measured on
the family's first build (`evidence/bf16-prefill-738e1eb2`; the 131,132-token prompt on the 262,144-token setting,
`evidence/bf16-prefill-256k`): a 131k prompt does not fit the default 131,072-token window, which refuses it by name.
A multi-turn chat's third turn, resumed from 5,493 of its 5,649 tokens, starts in 0.35 s against 4.4 s cold
(`evidence/bf16-speed`; 0.36 s with the drafter).

## Performance options

Each keeps the replies' bits; the numbers are this Mac's (`evidence/`).

- **Streams (`parallel`).** One undrafted stream decodes at 16.3 tokens/s, bound by reading the weights each step.
  Two streams give 28-32 tokens/s in all (67 ms a round against 62), each reply equal to its run alone; the first
  token moves from 0.23-0.26 s to 0.34-0.39 s. The engine's row kernel holds up to 32 rows a launch; the caches set the count.
- **Weights.** The release's bf16 is exact and is the only form served; no quantized conversion is offered.
- **Context and memory (`context`, `load_limit_gb`).** 208 KiB a token: 150 GB holds two 131,072-token streams, 192 GB
  one of 524,288. Cold prompts run 1.1-1.3k tokens/s to 33k tokens, 866 at 61k and 563 at 131k.
- **Prompt cache (`prompt_cache_gib`).** A resumed turn starts in 0.35 s against 4.4 s cold, with the same reply;
  kept states give way to streams that need the room.
- **Drafter (`drafter`).** Nanbeige4.1-3B's bf16 release proposing the next tokens, on by default: 99-114
  tokens/s on the harness's ruler against 16.3 undrafted, 21-41 on ordinary greedy chats, replies identical; two
  streams together 45-59 on the ruler against 28-32, 36.1 against 29.0 on fresh prompts
  ([below](#drafter-nanbeige41-3b)). With `drafter = false` the lane core's copy drafts alone accept 0-30% of their
  tokens on the harness's prompts and give no measurable decode gain (drafted 16 tokens/s, undrafted 16).
- **Thinking.** On by default (the template's); it lengthens replies, not the cost a token; prompt speed with it on is
  within 4% of off.
- **Not offered.** No int8 or fp8 cache, no fast-math, no reduced-precision sums.

## Drafter: Nanbeige4.1-3B

The 33B's decode is bound by reading 62 GiB of weights each step. Speculative decoding has a smaller model propose
the next few tokens and the 33B verify them all in one window, which costs about as much as one step; every token
the 33B keeps is one it would have drawn itself, so replies are the same. [Nanbeige4.1-3B](https://huggingface.co/Nanbeige/Nanbeige4.1-3B)
is a plain Llama on the same tokenizer (the same 166,100 ids, merges, special tokens and end tokens), so the engine's
llama family serves it as the drafter beside the 33B:

- `drafter = true` serves its bf16 release as published (Nanbeige/Nanbeige4.1-3B @ 9c4555c3, 7.33 GiB;
  `--drafter ~/models/Nanbeige4.1-3B --drafter-bits 0`). Nothing is converted, and the engine refuses quantized
  drafters by name.
- Before each round the drafter syncs its own cache to the stream's context (an accepted draft stays, a rejected one
  is cut), then runs greedy steps chained on the GPU, 8.5-8.9 ms each. The depth follows measured rates: the stream's
  acceptance, the step time and the 33B's window time by rows; with two streams, each round is priced for both, and
  each stream drafts the depth that lands the most tokens a millisecond for the round. A request takes one drafter
  cache (64 KiB a token), made or grown to its stream's capacity when the stream is admitted, in the same memory budget
  as the 33B's caches, checked before anything is allocated; a freed cache keeps its tokens for the next request
  sharing its prefix, so a continued chat is not fed again, and gives way when a stream needs the room.

Checked in the engine's own tool, which decodes greedily through the drafter's windows: on eight chat prompts (128
tokens, thinking on and off, English and Chinese) and after a 1,500-token prompt (256 tokens) the drafted tokens
equal the plain greedy tokens exactly, at 21.1-40.7 tokens/s against 15.8-16.1, with 1.7-4.5 tokens landing a round
(the engine doc's table has each prompt). Served (engine 50fe245a, no other model loaded):

| | undrafted | drafted |
|---|---:|---:|
| one stream, the ruler's prose / code / structured / list (256 tokens) | 16.3 / 16.2 / 16.3 / 16.3 tok/s | 113.9 / 108.3 / 98.6 / 98.9 |
| two streams together, the same | 31.7 / 27.8 / 31.4 / 31.6 | 44.6 / 59.0 / 54.4 / 55.9 |
| one stream's rounds | 61.7 ms, 1 token | 46.5 ms, 4.9 tokens (85% of drafts kept) |
| eight fresh prompts sent two at a time (greedy, 256 tokens, thinking off) | 26.6-30.1 (29.0) | 30.1-40.9 (36.1) |
| the suite's drafted vs undrafted check, one stream | 16 tok/s | 29 tok/s |
| first token, short prompt / resumed turn | 0.23-0.26 s / 0.35 s | 0.23-0.26 s / 0.36 s |

The ruler's texts repeat themselves, which the drafter continues almost whole; ordinary chats land 1.7-4.5 tokens a
round for 1.3-2.6x with greedy draws. Two streams share each round: every reply keeps its plain solo token ids
(streams joining, leaving, queued, refused and cancelled), drafting stays on in shared rounds, and the pair still runs
faster than undrafted on fresh prompts (25% on average). Every quick-plan check passes with the drafter
(`evidence/draft-quick`); the first token keeps its time, since the drafter reads the prompt after the model's own
prompt pass. Sampled decode speed was not measured with the bf16 drafter; seeded sampled replies equal the
undrafted ones (9/9).

## Notes

- No quality suite was run on this recipe; the checks above are exactness against the release's own code, the
  harness's serving checks and its speed rulers.
- `logprobs` requests are refused by name: the release's tokenizer decodes through a `Strip` step, which the engine's
  tokenizer does not count as token-local, so the server cannot give a token's bytes on its own.
- The family refuses what it has not been checked on: tied heads, head sizes other than 128, Llama 3's scaled RoPE,
  biased projections, and quantized checkpoints.

## License

The recipe is under the [MIT License](LICENSE). Nanbeige4.3-33B and Nanbeige4.1-3B are published under Apache 2.0 on
Hugging Face ([33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B), [3B](https://huggingface.co/Nanbeige/Nanbeige4.1-3B)).
