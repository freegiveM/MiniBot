from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluator import DEFAULT_ARTIFACT_PATH, DEFAULT_REAL_ARTIFACT_PATH, DEFAULT_WORKSPACE_ROOT, run_fixed_benchmark
from .metrics import (
    DEFAULT_CONTEXT_ARTIFACT_PATH,
    DEFAULT_HARNESS_ARTIFACT_PATH,
    DEFAULT_METHODOLOGY_REPORT_PATH,
    DEFAULT_MEMORY_ARTIFACT_PATH,
    DEFAULT_RECOVERY_ARTIFACT_PATH,
    DEFAULT_REAL_HARNESS_ARTIFACT_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_RETRIEVAL_ARTIFACT_PATH,
    write_benchmark_core_report,
    write_benchmark_methodology_report,
)
from .model_providers import (
    API_FORMAT_ANTHROPIC,
    API_FORMAT_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_DEEPSEEK,
    PROVIDER_FAKE,
    PROVIDER_HTTP,
    PROVIDER_OPENAI,
    ProviderConfigurationError,
    build_model_client_from_config,
    resolve_provider_config,
)
from .prompt_cache import PROMPT_CACHE_MODES, PROMPT_CACHE_RETENTIONS
from .repl import run_repl
from .runtime import MiniBot, SessionStore
from .workspace import WorkspaceContext


MINIBOT_BANNER = r"""MiniBot
 .---.
 / o o \
 \  ^  /
  `---'
"""
COMMANDS = frozenset({"benchmark", "metrics", "repl"})
APPROVAL_CHOICES = ("ask", "auto", "deny_risky", "never")
MODEL_PROVIDER_CHOICES = (PROVIDER_FAKE, PROVIDER_HTTP, PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_DEEPSEEK)
API_FORMAT_CHOICES = (API_FORMAT_OPENAI, API_FORMAT_ANTHROPIC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minibot",
        description=MINIBOT_BANNER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Commands:\n"
            "  minibot \"message\"        Run a single MiniBot task.\n"
            "  minibot repl             Start an interactive MiniBot REPL.\n"
            "  minibot benchmark         Run the fixed benchmark harness.\n"
            "  minibot metrics           Generate the benchmark core report."
        ),
    )
    parser.add_argument("message", nargs="*", help="Task message for the agent.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument("--approval", choices=APPROVAL_CHOICES, default="ask", help="Approval policy for risky tools.")
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool steps.")
    parser.add_argument("--model-provider", choices=MODEL_PROVIDER_CHOICES, default=None, help="Model provider.")
    parser.add_argument("--api-format", choices=API_FORMAT_CHOICES, default=None, help="HTTP provider wire format.")
    parser.add_argument("--model-name", default=None, help="Provider model name.")
    parser.add_argument("--base-url", default=None, help="Provider endpoint URL.")
    parser.add_argument("--api-key-env", default=None, help="Environment variable or .env key containing the API key.")
    parser.add_argument("--env-file", default=".env", help="Provider .env file path.")
    parser.add_argument("--prompt-cache", choices=PROMPT_CACHE_MODES, default=None, help="Provider prompt cache mode.")
    parser.add_argument("--prompt-cache-retention", choices=PROMPT_CACHE_RETENTIONS, default=None, help="Provider prompt cache retention.")
    parser.add_argument(
        "--fake-response",
        default="<final>MiniBot scaffold is running.</final>",
        help="Fake model response for smoke runs.",
    )
    return parser


def build_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minibot benchmark", description="Run MiniBot fixed benchmark tasks.")
    parser.add_argument("--cwd", default=".", help="Project root used to resolve relative benchmark paths.")
    parser.add_argument("--benchmark-path", default="benchmarks/coding_tasks.json")
    parser.add_argument("--artifact-path", default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
    parser.add_argument("--real", action="store_true", help="Run a real-model benchmark instead of scripted mock.")
    parser.add_argument("--model-provider", choices=MODEL_PROVIDER_CHOICES, default=None, help="Model provider.")
    parser.add_argument("--api-format", choices=API_FORMAT_CHOICES, default=None, help="HTTP provider wire format.")
    parser.add_argument("--model-name", default=None, help="Provider model name.")
    parser.add_argument("--base-url", default=None, help="Provider endpoint URL.")
    parser.add_argument("--api-key-env", default=None, help="Environment variable or .env key containing the API key.")
    parser.add_argument("--env-file", default=".env", help="Provider .env file path.")
    parser.add_argument("--prompt-cache", choices=PROMPT_CACHE_MODES, default=None, help="Provider prompt cache mode.")
    parser.add_argument("--prompt-cache-retention", choices=PROMPT_CACHE_RETENTIONS, default=None, help="Provider prompt cache retention.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Provider decoding temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum provider output tokens.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit benchmark tasks.")
    parser.add_argument("--max-estimated-cost", type=float, default=None, help="Stop after this estimated USD cost.")
    parser.add_argument("--dry-run", action="store_true", help="Write a planned benchmark artifact without running tasks.")
    return parser


def build_metrics_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minibot metrics", description="Generate MiniBot benchmark reports.")
    parser.add_argument("--cwd", default=".", help="Project root used to resolve relative artifact paths.")
    parser.add_argument("--harness-artifact-path", default=str(DEFAULT_HARNESS_ARTIFACT_PATH))
    parser.add_argument("--real-harness-artifact-path", default=str(DEFAULT_REAL_HARNESS_ARTIFACT_PATH))
    parser.add_argument("--context-artifact-path", default=str(DEFAULT_CONTEXT_ARTIFACT_PATH))
    parser.add_argument("--memory-artifact-path", default=str(DEFAULT_MEMORY_ARTIFACT_PATH))
    parser.add_argument("--recovery-artifact-path", default=str(DEFAULT_RECOVERY_ARTIFACT_PATH))
    parser.add_argument("--retrieval-artifact-path", default=str(DEFAULT_RETRIEVAL_ARTIFACT_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument(
        "--methodology-report",
        action="store_true",
        help=f"Write the Stage 20 methodology report instead of the core report. Default path: {DEFAULT_METHODOLOGY_REPORT_PATH}",
    )
    return parser


def build_repl_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minibot repl", description="Start an interactive MiniBot REPL.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument("--resume", default=None, help="Resume an existing MiniBot session id.")
    parser.add_argument(
        "--approval",
        choices=APPROVAL_CHOICES,
        default="ask",
        help="Approval policy for risky tools.",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool steps per REPL turn.")
    parser.add_argument("--model-provider", choices=MODEL_PROVIDER_CHOICES, default=None, help="Model provider.")
    parser.add_argument("--api-format", choices=API_FORMAT_CHOICES, default=None, help="HTTP provider wire format.")
    parser.add_argument("--model-name", default=None, help="Provider model name.")
    parser.add_argument("--base-url", default=None, help="Provider endpoint URL.")
    parser.add_argument("--api-key-env", default=None, help="Environment variable or .env key containing the API key.")
    parser.add_argument("--env-file", default=".env", help="Provider .env file path.")
    parser.add_argument("--prompt-cache", choices=PROMPT_CACHE_MODES, default=None, help="Provider prompt cache mode.")
    parser.add_argument("--prompt-cache-retention", choices=PROMPT_CACHE_RETENTIONS, default=None, help="Provider prompt cache retention.")
    parser.add_argument(
        "--fake-response",
        default="<final>MiniBot scaffold is running.</final>",
        help="Fake model response for smoke runs.",
    )
    return parser


def build_agent_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minibot",
        description=MINIBOT_BANNER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("message", nargs="*", help="Task message for the agent.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument(
        "--approval",
        choices=APPROVAL_CHOICES,
        default="ask",
        help="Approval policy for risky tools.",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool steps.")
    parser.add_argument("--model-provider", choices=MODEL_PROVIDER_CHOICES, default=None, help="Model provider.")
    parser.add_argument("--api-format", choices=API_FORMAT_CHOICES, default=None, help="HTTP provider wire format.")
    parser.add_argument("--model-name", default=None, help="Provider model name.")
    parser.add_argument("--base-url", default=None, help="Provider endpoint URL.")
    parser.add_argument("--api-key-env", default=None, help="Environment variable or .env key containing the API key.")
    parser.add_argument("--env-file", default=".env", help="Provider .env file path.")
    parser.add_argument("--prompt-cache", choices=PROMPT_CACHE_MODES, default=None, help="Provider prompt cache mode.")
    parser.add_argument("--prompt-cache-retention", choices=PROMPT_CACHE_RETENTIONS, default=None, help="Provider prompt cache retention.")
    parser.add_argument(
        "--fake-response",
        default="<final>MiniBot scaffold is running.</final>",
        help="Fake model response for smoke runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        build_parser().print_help()
        return 0

    command, command_argv = _split_command(argv)
    if command == "benchmark":
        return run_benchmark_command(command_argv)
    if command == "metrics":
        return run_metrics_command(command_argv)
    if command == "repl":
        return run_repl_command(command_argv)
    return run_agent_command(argv)


def run_agent_command(argv: list[str]) -> int:
    parser = build_agent_parser()
    parsed = _parse_or_exit_code(parser, argv)
    if isinstance(parsed, int):
        return parsed
    if parsed.max_steps <= 0:
        parser.print_usage(sys.stderr)
        print("minibot: error: --max-steps must be positive", file=sys.stderr)
        return 2

    message = " ".join(parsed.message).strip()
    if not message:
        parser.print_help()
        return 0

    cwd = Path(parsed.cwd).resolve()
    if not cwd.exists() or not cwd.is_dir():
        print(f"minibot: --cwd does not exist or is not a directory: {cwd}", file=sys.stderr)
        return 2
    try:
        model = build_model_client(
            cwd=cwd,
            env_file=parsed.env_file,
            model_provider=parsed.model_provider,
            api_format=parsed.api_format,
            model_name=parsed.model_name,
            base_url=parsed.base_url,
            api_key_env=parsed.api_key_env,
            prompt_cache=parsed.prompt_cache,
            prompt_cache_retention=parsed.prompt_cache_retention,
            fake_response=parsed.fake_response,
        )
    except ProviderConfigurationError as exc:
        print(f"minibot: provider configuration error: {exc}", file=sys.stderr)
        return 2
    workspace = WorkspaceContext.build(cwd)
    state_root = Path(workspace.repo_root) / ".minibot"
    session_store = SessionStore(state_root / "sessions")
    agent = MiniBot(
        model_client=model,
        workspace=workspace,
        session_store=session_store,
        approval_policy=parsed.approval,
        max_steps=parsed.max_steps,
    )
    print(agent.ask(message))
    return 0


def run_benchmark_command(argv: list[str]) -> int:
    parser = build_benchmark_parser()
    parsed = _parse_or_exit_code(parser, argv)
    if isinstance(parsed, int):
        return parsed
    cwd = Path(parsed.cwd).resolve()
    real = bool(parsed.real or (parsed.model_provider and parsed.model_provider != PROVIDER_FAKE))
    artifact_path = parsed.artifact_path
    if real and artifact_path == str(DEFAULT_ARTIFACT_PATH):
        artifact_path = str(DEFAULT_REAL_ARTIFACT_PATH)
    try:
        artifact = run_fixed_benchmark(
            benchmark_path=_resolve_cli_path(cwd, parsed.benchmark_path),
            artifact_path=_resolve_cli_path(cwd, artifact_path),
            workspace_root=_resolve_cli_path(cwd, parsed.workspace_root),
            real=real,
            model_provider=parsed.model_provider,
            api_format=parsed.api_format,
            model_name=parsed.model_name,
            base_url=parsed.base_url,
            api_key_env=parsed.api_key_env,
            env_file=_resolve_cli_path(cwd, parsed.env_file),
            prompt_cache=parsed.prompt_cache,
            prompt_cache_retention=parsed.prompt_cache_retention,
            temperature=parsed.temperature,
            max_new_tokens=parsed.max_new_tokens,
            max_tasks=parsed.max_tasks,
            max_estimated_cost=parsed.max_estimated_cost,
            dry_run=parsed.dry_run,
        )
    except ProviderConfigurationError as exc:
        print(f"minibot: provider configuration error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(artifact["summary"], sort_keys=True, ensure_ascii=False))
    return 0


def run_metrics_command(argv: list[str]) -> int:
    parser = build_metrics_parser()
    parsed = _parse_or_exit_code(parser, argv)
    if isinstance(parsed, int):
        return parsed
    cwd = Path(parsed.cwd).resolve()
    report_path = _resolve_cli_path(
        cwd,
        str(DEFAULT_METHODOLOGY_REPORT_PATH) if parsed.methodology_report and parsed.report_path == str(DEFAULT_REPORT_PATH) else parsed.report_path,
    )
    if parsed.methodology_report:
        write_benchmark_methodology_report(
            report_path=report_path,
            mock_harness_artifact_path=_resolve_cli_path(cwd, parsed.harness_artifact_path),
            real_harness_artifact_path=_resolve_cli_path(cwd, parsed.real_harness_artifact_path),
            context_artifact_path=_resolve_cli_path(cwd, parsed.context_artifact_path),
            memory_artifact_path=_resolve_cli_path(cwd, parsed.memory_artifact_path),
            recovery_artifact_path=_resolve_cli_path(cwd, parsed.recovery_artifact_path),
            retrieval_artifact_path=_resolve_cli_path(cwd, parsed.retrieval_artifact_path),
        )
    else:
        write_benchmark_core_report(
            report_path=report_path,
            harness_artifact_path=_resolve_cli_path(cwd, parsed.harness_artifact_path),
            context_artifact_path=_resolve_cli_path(cwd, parsed.context_artifact_path),
            memory_artifact_path=_resolve_cli_path(cwd, parsed.memory_artifact_path),
            recovery_artifact_path=_resolve_cli_path(cwd, parsed.recovery_artifact_path),
            retrieval_artifact_path=_resolve_cli_path(cwd, parsed.retrieval_artifact_path),
        )
    print(str(report_path))
    return 0


def run_repl_command(argv: list[str]) -> int:
    parser = build_repl_parser()
    parsed = _parse_or_exit_code(parser, argv)
    if isinstance(parsed, int):
        return parsed
    if parsed.max_steps <= 0:
        parser.print_usage(sys.stderr)
        print("minibot: error: --max-steps must be positive", file=sys.stderr)
        return 2

    cwd = Path(parsed.cwd).resolve()
    if not cwd.exists() or not cwd.is_dir():
        print(f"minibot: --cwd does not exist or is not a directory: {cwd}", file=sys.stderr)
        return 2
    try:
        model = build_model_client(
            cwd=cwd,
            env_file=parsed.env_file,
            model_provider=parsed.model_provider,
            api_format=parsed.api_format,
            model_name=parsed.model_name,
            base_url=parsed.base_url,
            api_key_env=parsed.api_key_env,
            prompt_cache=parsed.prompt_cache,
            prompt_cache_retention=parsed.prompt_cache_retention,
            fake_response=parsed.fake_response,
        )
    except ProviderConfigurationError as exc:
        print(f"minibot: provider configuration error: {exc}", file=sys.stderr)
        return 2

    return run_repl(
        cwd=cwd,
        model_client=model,
        approval_policy=parsed.approval,
        max_steps=parsed.max_steps,
        resume=parsed.resume,
        input_stream=sys.stdin,
        output_stream=sys.stdout,
        error_stream=sys.stderr,
    )


def build_model_client(
    *,
    cwd: str | Path = ".",
    env_file: str | Path = ".env",
    model_provider: str | None = None,
    api_format: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    prompt_cache: str | None = None,
    prompt_cache_retention: str | None = None,
    fake_response: str = "<final>MiniBot scaffold is running.</final>",
):
    config = resolve_provider_config(
        cwd=cwd,
        env_file=env_file,
        model_provider=model_provider,
        api_format=api_format,
        model_name=model_name,
        base_url=base_url,
        api_key_env=api_key_env,
        prompt_cache=prompt_cache,
        prompt_cache_retention=prompt_cache_retention,
    )
    return build_model_client_from_config(config, fake_response=fake_response)


def _split_command(argv: list[str]) -> tuple[str, list[str]]:
    if argv[0] in COMMANDS:
        return argv[0], argv[1:]
    if len(argv) >= 3 and argv[0] == "--cwd" and argv[2] in COMMANDS:
        return argv[2], [argv[0], argv[1], *argv[3:]]
    return "", argv


def _resolve_cli_path(cwd: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return cwd / path


def _parse_or_exit_code(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace | int:
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
