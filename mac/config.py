"""Settings: recipe.toml's defaults, local.toml's overrides for this Mac, then environment variables in upper case."""

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    model_repo: str = ""
    model_revision: str = ""
    served_name: str = ""
    variant: str = "bf16"
    model_dir: str = ""
    model_dir_8bit: str = ""
    model_dir_4bit: str = ""
    drafter_repo: str = ""
    drafter_revision: str = ""
    drafter_dir: str = ""
    drafter_dir_4bit: str = ""
    tensorfold_src: str = ""
    tensorfold_sha: str = ""
    zig: str = "zig"
    zig_cpu: str = "native"
    binary: str = ""
    host: str = "127.0.0.1"
    port: int = 8896
    tf_cache: str = "~/.cache/nanbeige43-33b"
    context: int = 131072
    parallel: int = 2
    max_tokens: int = 32768
    temperature: str = ""
    top_p: str = ""
    top_k: str = ""
    prompt_cache_gib: float = 8
    load_limit_gb: float = 150
    drafts: bool = True
    drafter: bool = True
    keep_warm: int = 86400
    api_key: str = ""
    extra_args: str = ""

    @staticmethod
    def path(p: str) -> Path:
        return Path(os.path.expanduser(p))

    @property
    def model(self) -> Path:
        """The served checkpoint: the release (bf16), or the conversion `python3 -m mac convert` writes for the variant."""
        if self.variant == "8bit":
            return self.path(self.model_dir_8bit)
        if self.variant == "4bit":
            return self.path(self.model_dir_4bit)
        return self.path(self.model_dir)

    @property
    def drafter_model(self) -> Path:
        """The drafter the server loads: the 4-bit conversion `python3 -m mac convert --drafter` writes."""
        return self.path(self.drafter_dir_4bit)

    @property
    def cache(self) -> Path:
        return self.path(self.tf_cache)

    @property
    def worktree(self) -> Path:
        return self.cache / "engine" / self.tensorfold_sha[:12]

    @property
    def built_binary(self) -> Path:
        return self.worktree / "zig-out" / "native" / "bin" / "tensorfold-native"

    @property
    def built_grammar(self) -> Path:
        """libtfgrammar (xgrammar), which the server loads beside its binary (../lib) at the first structured request."""
        return self.worktree / "zig-out" / "native" / "lib" / "libtfgrammar.dylib"

    @property
    def run_binary(self) -> Path:
        return self.path(self.binary) if self.binary else self.built_binary

    @property
    def pid_file(self) -> Path:
        return self.cache / "server.pid"

    @property
    def log_file(self) -> Path:
        return self.cache / "server.log"

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}/v1"

    def serve_args(self) -> list[str]:
        a = [str(self.run_binary), "serve", "--host", self.host, "--port", str(self.port), "--name", self.served_name,
             "--context", str(self.context), "--max-tokens", str(self.max_tokens), "--parallel", str(self.parallel),
             "--keep-warm", str(self.keep_warm), "--prompt-cache-gib", f"{self.prompt_cache_gib:g}", "--no-update-check"]
        for flag, value in (("--temperature", self.temperature), ("--top-p", self.top_p), ("--top-k", self.top_k)):
            if value:
                a += [flag, value]
        if not self.drafts:
            a.append("--no-drafts")
        elif self.drafter:
            a += ["--drafter", str(self.drafter_model), "--drafter-bits", "4"]
        if self.api_key:
            a += ["--api-key", self.api_key]
        a += self.extra_args.split()
        a.append(str(self.model))
        return a

    def server_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.load_limit_gb > 0:
            env["LLAMA_LOAD_LIMIT_GB"] = f"{self.load_limit_gb:g}"
        return env


def _coerce(kind: type, raw):
    if kind is bool:
        return raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
    return kind(raw)


def load(local_file: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    merged: dict = {}
    for f in (ROOT / "recipe.toml", local_file or ROOT / "local.toml"):
        if f.exists():
            for section in tomllib.loads(f.read_text()).values():
                merged.update(section)
    s = Settings()
    for fld in fields(Settings):
        kind = type(getattr(s, fld.name))
        if fld.name in merged:
            setattr(s, fld.name, _coerce(kind, merged[fld.name]))
        if fld.name.upper() in env:
            setattr(s, fld.name, _coerce(kind, env[fld.name.upper()]))
    return s


def validate(s: Settings) -> list[str]:
    """The settings' problems, worded for the user (empty: none)."""
    out = []
    if not s.binary and not s.tensorfold_sha:
        out.append("no engine: set tensorfold_sha (local.toml) or binary")
    if s.variant not in ("bf16", "8bit", "4bit"):
        out.append("variant must be bf16 (the release as published), 8bit or 4bit (its MLX affine conversions)")
    elif not (s.model / "config.json").exists():
        out.append(f"no checkpoint at {s.model}: hf download {s.model_repo} --revision {s.model_revision} --local-dir {s.model_dir}"
                   + (" then python3 -m mac convert" if s.variant != "bf16" else ""))
    if s.drafts and s.drafter and not (s.drafter_model / "config.json").exists():
        out.append(f"no drafter at {s.drafter_model}: hf download {s.drafter_repo} --revision {s.drafter_revision} --local-dir {s.drafter_dir}"
                   " then python3 -m mac convert --drafter (or drafter = false for the lane engine's copy drafts alone)")
    if s.parallel < 1 or s.parallel > 16:
        out.append("parallel must be 1 to 16")
    if s.context < 1 or s.context > 524288:
        out.append("context must be 1 to 524288 (the release's max_position_embeddings)")
    if not s.run_binary.exists():
        out.append(f"no engine binary at {s.run_binary}: python3 -m mac build")
    elif not s.binary and not s.built_grammar.exists():
        out.append(f"no libtfgrammar beside the engine ({s.built_grammar}): python3 -m mac build")
    return out
