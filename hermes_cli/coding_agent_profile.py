"""CLI profile isolation helper for standalone coding agents.

This module intentionally manages only non-secret metadata and environment
selection. It does not copy, print, parse, or bridge OAuth tokens. Authenticate
individual profiles later with the underlying CLI while the profile environment
is active.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

STATE_VERSION = 1
ROOT_ENV = "CODING_AGENT_PROFILES_HOME"
DEFAULT_ROOT = Path.home() / ".coding-agent-profiles"
SUPPORTED_PROVIDERS = ("codex", "claude")
DEFAULT_POOL = ("codex/main", "codex/sub1", "claude/main", "claude/sub1")
TASK_POLICIES = {
    "implementation": ("codex/main", "codex/sub1", "claude/main", "claude/sub1"),
    "coding": ("codex/main", "codex/sub1", "claude/main", "claude/sub1"),
    "fast": ("codex/main", "codex/sub1", "claude/main", "claude/sub1"),
    "review": ("claude/main", "claude/sub1", "codex/main", "codex/sub1"),
    "planning": ("claude/main", "claude/sub1", "codex/main", "codex/sub1"),
    "large-context": ("claude/main", "claude/sub1", "codex/main", "codex/sub1"),
    "default": DEFAULT_POOL,
}
DEFAULT_PROVIDER_COMMANDS = {
    "codex": "codex exec {prompt:q}",
    "claude": "claude -p {prompt:q}",
}
RETRYABLE_PATTERNS = (
    "usage limit",
    "rate limit",
    "quota",
    "too many requests",
    "retry-after",
    "retry after",
    "billing hard limit",
    "temporarily unavailable",
)
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class Profile:
    provider: str
    label: str
    path: Path
    env: dict[str, str]


def profiles_root() -> Path:
    raw = os.environ.get(ROOT_ENV)
    return Path(raw).expanduser() if raw else DEFAULT_ROOT


def state_path(root: Path | None = None) -> Path:
    return (root or profiles_root()) / "state.json"


def _now() -> int:
    return int(time.time())


def validate_provider(provider: str) -> str:
    if provider not in SUPPORTED_PROVIDERS:
        raise SystemExit(f"unsupported provider: {provider}")
    return provider


def validate_label(label: str) -> str:
    if not _LABEL_RE.match(label):
        raise SystemExit(
            "invalid label: use 1-64 chars, starting with alnum, "
            "then alnum/dot/underscore/hyphen"
        )
    return label


def profile_path(provider: str, label: str, root: Path | None = None) -> Path:
    validate_provider(provider)
    validate_label(label)
    root = root or profiles_root()
    if provider == "codex":
        return root / "codex" / label
    if provider == "claude":
        return root / "claude" / label / "home"
    raise AssertionError(provider)


def profile_env(provider: str, label: str, root: Path | None = None) -> dict[str, str]:
    path = profile_path(provider, label, root)
    if provider == "codex":
        return {"CODEX_HOME": str(path)}
    if provider == "claude":
        return {"HOME": str(path)}
    raise AssertionError(provider)


def load_state(root: Path | None = None) -> dict:
    path = state_path(root)
    if not path.exists():
        return {"version": STATE_VERSION, "profiles": {}}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"state file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"state file must contain a JSON object: {path}")
    data.setdefault("version", STATE_VERSION)
    data.setdefault("profiles", {})
    return data


def save_state(data: dict, root: Path | None = None) -> None:
    path = state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def profile_key(provider: str, label: str) -> str:
    return f"{provider}/{label}"


def parse_profile_ref(ref: str) -> tuple[str, str]:
    if "/" not in ref:
        raise SystemExit(f"profile must be provider/label, got: {ref}")
    provider, label = ref.split("/", 1)
    return validate_provider(provider), validate_label(label)


def ensure_profile_ref(ref: str, *, root: Path | None = None) -> Profile:
    provider, label = parse_profile_ref(ref)
    return ensure_profile(provider, label, root=root)


def ensure_profile(provider: str, label: str, *, root: Path | None = None) -> Profile:
    validate_provider(provider)
    validate_label(label)
    root = root or profiles_root()
    path = profile_path(provider, label, root)
    path.mkdir(parents=True, exist_ok=True)
    if provider == "claude":
        (path / ".claude").mkdir(parents=True, exist_ok=True)
    data = load_state(root)
    key = profile_key(provider, label)
    profiles = data.setdefault("profiles", {})
    now = _now()
    existing = profiles.get(key) if isinstance(profiles.get(key), dict) else {}
    profiles[key] = {
        **existing,
        "provider": provider,
        "label": label,
        "path": str(path),
        "env": profile_env(provider, label, root),
        "created_at": existing.get("created_at", now),
        "updated_at": now,
    }
    save_state(data, root)
    return Profile(provider=provider, label=label, path=path, env=profile_env(provider, label, root))


def iter_profiles(provider: str | None = None, *, root: Path | None = None) -> list[Profile]:
    root = root or profiles_root()
    data = load_state(root)
    result: list[Profile] = []
    for item in data.get("profiles", {}).values():
        if not isinstance(item, dict):
            continue
        item_provider = item.get("provider")
        label = item.get("label")
        if item_provider not in SUPPORTED_PROVIDERS or not isinstance(label, str):
            continue
        if provider and item_provider != provider:
            continue
        result.append(
            Profile(
                provider=item_provider,
                label=label,
                path=Path(item.get("path") or profile_path(item_provider, label, root)),
                env=profile_env(item_provider, label, root),
            )
        )
    return sorted(result, key=lambda p: (p.provider, p.label))


def auth_status(profile: Profile) -> str:
    """Return a conservative non-secret status based on auth file presence."""
    if profile.provider == "codex":
        candidates = [profile.path / "auth.json", profile.path / ".codex" / "auth.json"]
    elif profile.provider == "claude":
        candidates = [
            profile.path / ".claude.json",
            profile.path / ".claude" / ".credentials.json",
            profile.path / ".config" / "claude" / ".credentials.json",
        ]
    else:
        candidates = []
    return "auth_file_present" if any(path.exists() for path in candidates) else "not_authenticated"


def merged_env(profile: Profile) -> dict[str, str]:
    env = os.environ.copy()
    env.update(profile.env)
    return env


def print_profile(profile: Profile, *, json_output: bool = False) -> None:
    payload = {
        "provider": profile.provider,
        "label": profile.label,
        "path": str(profile.path),
        "env": profile.env,
        "auth_status": auth_status(profile),
    }
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"{profile.provider}/{profile.label}\t{payload['auth_status']}\t"
            f"path={profile.path}"
        )


def emit_env(profile: Profile, *, shell: str) -> None:
    if shell not in {"sh", "json"}:
        raise SystemExit(f"unsupported shell: {shell}")
    if shell == "json":
        print(json.dumps(profile.env, indent=2, sort_keys=True))
        return
    for key, value in profile.env.items():
        print(f"export {key}={shlex.quote(value)}")


def login_hint(profile: Profile) -> str:
    if profile.provider == "codex":
        return f"CODEX_HOME={shlex.quote(str(profile.path))} codex login"
    if profile.provider == "claude":
        return f"HOME={shlex.quote(str(profile.path))} claude auth login"
    raise AssertionError(profile.provider)


def run_with_profile(profile: Profile, command: Sequence[str]) -> int:
    if not command:
        raise SystemExit("missing command after --")
    return subprocess.call(list(command), env=merged_env(profile))


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stdout or "") + (result.stderr or "")


def is_retryable_failure(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in RETRYABLE_PATTERNS)


def rotate_profiles(provider: str, labels: Iterable[str], command: Sequence[str]) -> int:
    if not command:
        raise SystemExit("missing command after --")
    selected = [ensure_profile(provider, label) for label in labels]
    if not selected:
        raise SystemExit("no profiles selected")
    last_code = 1
    for index, profile in enumerate(selected):
        print(f"coding-agent-profile: trying {profile.provider}/{profile.label}", file=sys.stderr)
        result = subprocess.run(
            list(command),
            env=merged_env(profile),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        last_code = result.returncode
        if result.returncode == 0:
            print(f"coding-agent-profile: {profile.provider}/{profile.label} succeeded", file=sys.stderr)
            return 0
        output = _combined_output(result)
        if index < len(selected) - 1 and is_retryable_failure(output):
            next_profile = selected[index + 1]
            print(
                f"coding-agent-profile: {profile.provider}/{profile.label} failed with retryable limit; "
                f"retrying with {next_profile.provider}/{next_profile.label}",
                file=sys.stderr,
            )
            continue
        print(
            f"coding-agent-profile: {profile.provider}/{profile.label} failed; not retrying",
            file=sys.stderr,
        )
        return result.returncode
    return last_code


def split_profile_refs(raw: str | None, *, task: str = "default") -> list[str]:
    if raw:
        refs = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        refs = list(TASK_POLICIES.get(task, TASK_POLICIES["default"]))
    # Validate eagerly and preserve order.
    for ref in refs:
        parse_profile_ref(ref)
    return refs


def recommended_profiles(task: str) -> tuple[str, ...]:
    return tuple(TASK_POLICIES.get(task, TASK_POLICIES["default"]))


def format_provider_command(template: str, *, profile: Profile, prompt: str) -> list[str]:
    rendered = template.replace("{provider}", profile.provider).replace("{label}", profile.label)
    rendered = rendered.replace("{profile}", profile_key(profile.provider, profile.label))
    rendered = rendered.replace("{prompt:q}", shlex.quote(prompt))
    rendered = rendered.replace("{prompt}", prompt)
    return shlex.split(rendered)


def mixed_rotate_profiles(
    refs: Iterable[str],
    *,
    prompt: str,
    codex_command: str | None = None,
    claude_command: str | None = None,
    root: Path | None = None,
    dry_run: bool = False,
) -> int:
    templates = {
        "codex": codex_command or DEFAULT_PROVIDER_COMMANDS["codex"],
        "claude": claude_command or DEFAULT_PROVIDER_COMMANDS["claude"],
    }
    profiles = [ensure_profile_ref(ref, root=root) for ref in refs]
    if not profiles:
        raise SystemExit("no profiles selected")
    if dry_run:
        for profile in profiles:
            command = format_provider_command(templates[profile.provider], profile=profile, prompt=prompt)
            print(json.dumps({
                "profile": profile_key(profile.provider, profile.label),
                "auth_status": auth_status(profile),
                "env": profile.env,
                "command": command,
            }, sort_keys=True))
        return 0
    last_code = 1
    for index, profile in enumerate(profiles):
        ref = profile_key(profile.provider, profile.label)
        command = format_provider_command(templates[profile.provider], profile=profile, prompt=prompt)
        print(f"coding-agent-profile: trying {ref}", file=sys.stderr)
        result = subprocess.run(
            command,
            env=merged_env(profile),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        last_code = result.returncode
        if result.returncode == 0:
            print(f"coding-agent-profile: {ref} succeeded", file=sys.stderr)
            return 0
        if index < len(profiles) - 1 and is_retryable_failure(_combined_output(result)):
            next_profile = profiles[index + 1]
            print(
                f"coding-agent-profile: {ref} failed with retryable limit; "
                f"retrying with {profile_key(next_profile.provider, next_profile.label)}",
                file=sys.stderr,
            )
            continue
        print(f"coding-agent-profile: {ref} failed; not retrying", file=sys.stderr)
        return result.returncode
    return last_code


def _split_labels(raw: str | None, provider: str) -> list[str]:
    if raw:
        return [validate_label(part.strip()) for part in raw.split(",") if part.strip()]
    profiles = iter_profiles(provider)
    return [profile.label for profile in profiles]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coding-agent-profile",
        description="Manage non-secret Codex/Claude CLI profile environments.",
    )
    parser.add_argument(
        "--root",
        help=f"Profile root (default: ${ROOT_ENV} or ~/.coding-agent-profiles)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_pool = sub.add_parser("init-pool", help="Create the default 4-account pool")
    init_pool.add_argument(
        "--profiles",
        default=",".join(DEFAULT_POOL),
        help="Comma-separated provider/label refs (default: codex/main,codex/sub1,claude/main,claude/sub1)",
    )
    init_pool.add_argument("--json", action="store_true")

    recommend = sub.add_parser("recommend", help="Recommend profile priority order for a task type")
    recommend.add_argument("--task", default="default", help="Task type: implementation, coding, fast, review, planning, large-context")
    recommend.add_argument("--profiles", help="Override comma-separated provider/label refs")
    recommend.add_argument("--json", action="store_true")

    run_policy = sub.add_parser("run-policy", help="Run a prompt with mixed Codex/Claude priority fallback")
    run_policy.add_argument("--task", default="default")
    run_policy.add_argument("--profiles", help="Comma-separated provider/label refs. Defaults to recommendation for --task")
    run_policy.add_argument("--prompt", required=True)
    run_policy.add_argument("--codex-command", help="Template for Codex profiles; supports {prompt:q}, {profile}, {provider}, {label}")
    run_policy.add_argument("--claude-command", help="Template for Claude profiles; supports {prompt:q}, {profile}, {provider}, {label}")
    run_policy.add_argument("--dry-run", action="store_true", help="Print planned commands without executing")

    for provider in SUPPORTED_PROVIDERS:
        provider_parser = sub.add_parser(provider, help=f"Manage {provider} profiles")
        actions = provider_parser.add_subparsers(dest="action", required=True)

        create = actions.add_parser("create", help="Create/update a profile directory")
        create.add_argument("label")
        create.add_argument("--json", action="store_true")

        list_cmd = actions.add_parser("list", help="List profiles")
        list_cmd.add_argument("--json", action="store_true")

        status = actions.add_parser("status", help="Show one profile status")
        status.add_argument("label")
        status.add_argument("--json", action="store_true")

        env = actions.add_parser("env", help="Print shell exports for a profile")
        env.add_argument("label")
        env.add_argument("--shell", choices=("sh", "json"), default="sh")

        login = actions.add_parser("login-command", help="Print the auth command to run later")
        login.add_argument("label")

        run = actions.add_parser("run", help="Run a command with a profile environment")
        run.add_argument("label")
        run.add_argument("argv", nargs=argparse.REMAINDER)

        rotate = actions.add_parser("rotate", help="Retry one provider command across profiles on quota/rate-limit failures")
        rotate.add_argument("--profiles", help="Comma-separated labels. Defaults to all profiles for provider.")
        rotate.add_argument("argv", nargs=argparse.REMAINDER)
    return parser


def normalize_command(command: Sequence[str]) -> list[str]:
    items = list(command)
    if items and items[0] == "--":
        return items[1:]
    return items


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser() if args.root else profiles_root()

    if args.command == "init-pool":
        profiles = [ensure_profile_ref(ref, root=root) for ref in split_profile_refs(args.profiles)]
        if args.json:
            print(json.dumps([
                {
                    "provider": p.provider,
                    "label": p.label,
                    "profile": profile_key(p.provider, p.label),
                    "path": str(p.path),
                    "env": p.env,
                    "auth_status": auth_status(p),
                }
                for p in profiles
            ], indent=2, sort_keys=True))
        else:
            for profile in profiles:
                print_profile(profile)
        return 0

    if args.command == "recommend":
        refs = split_profile_refs(args.profiles, task=args.task)
        if args.json:
            print(json.dumps({"task": args.task, "profiles": refs}, indent=2, sort_keys=True))
        else:
            print(" -> ".join(refs))
        return 0

    if args.command == "run-policy":
        refs = split_profile_refs(args.profiles, task=args.task)
        return mixed_rotate_profiles(
            refs,
            prompt=args.prompt,
            codex_command=args.codex_command,
            claude_command=args.claude_command,
            root=root,
            dry_run=args.dry_run,
        )

    provider = validate_provider(args.command)
    action = args.action

    if action == "create":
        profile = ensure_profile(provider, args.label, root=root)
        print_profile(profile, json_output=args.json)
        return 0
    if action == "list":
        profiles = iter_profiles(provider, root=root)
        if args.json:
            print(json.dumps([
                {
                    "provider": p.provider,
                    "label": p.label,
                    "path": str(p.path),
                    "env": p.env,
                    "auth_status": auth_status(p),
                }
                for p in profiles
            ], indent=2, sort_keys=True))
        else:
            for profile in profiles:
                print_profile(profile)
        return 0
    if action == "status":
        print_profile(ensure_profile(provider, args.label, root=root), json_output=args.json)
        return 0
    if action == "env":
        emit_env(ensure_profile(provider, args.label, root=root), shell=args.shell)
        return 0
    if action == "login-command":
        print(login_hint(ensure_profile(provider, args.label, root=root)))
        return 0
    if action == "run":
        profile = ensure_profile(provider, args.label, root=root)
        return run_with_profile(profile, normalize_command(args.argv))
    if action == "rotate":
        labels = _split_labels(args.profiles, provider)
        return rotate_profiles(provider, labels, normalize_command(args.argv))
    parser.error(f"unknown action: {action}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
