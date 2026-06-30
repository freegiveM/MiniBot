from __future__ import annotations

import hashlib
import json
import os
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


MAX_TOOL_OUTPUT = 4000
DOC_NAMES = ("AGENT.md", "AGENTS.md", "README.md", "pyproject.toml")
IGNORED_PATH_NAMES = {
    ".git",
    ".minibot",
    ".pico",
    "pico",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clip(text: object, limit: int = MAX_TOOL_OUTPUT) -> str:
    value = str(text)
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated {len(value) - limit} chars]"


@dataclass
class WorkspaceContext:
    cwd: str
    repo_root: str
    branch: str
    default_branch: str
    status: str
    recent_commits: list[str]
    project_docs: dict[str, str]

    @classmethod
    def build(cls, cwd: str | Path, repo_root_override: str | Path | None = None) -> "WorkspaceContext":
        cwd = Path(cwd).resolve()

        def run_git(args: list[str], fallback: str = "", *, git_cwd: Path | None = None, env: dict | None = None) -> str:
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=git_cwd or cwd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                    env=env,
                )
                return result.stdout.strip() or fallback
            except Exception:
                return fallback

        if repo_root_override is not None:
            repo_root = Path(repo_root_override).resolve()
            git_enabled = (repo_root / ".git").exists()
            git_cwd = repo_root
            git_env = os.environ.copy()
            git_env["GIT_CEILING_DIRECTORIES"] = str(repo_root.parent)
        else:
            repo_root = Path(run_git(["rev-parse", "--show-toplevel"], str(cwd))).resolve()
            git_enabled = True
            git_cwd = cwd
            git_env = None

        def git(args: list[str], fallback: str = "") -> str:
            if not git_enabled:
                return fallback
            return run_git(args, fallback, git_cwd=git_cwd, env=git_env)

        docs: dict[str, str] = {}
        for name in DOC_NAMES:
            path = repo_root / name
            if path.exists() and path.is_file():
                docs[name] = clip(path.read_text(encoding="utf-8", errors="replace"), 1200)

        default_branch = git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], "origin/main") or "origin/main"
        if default_branch.startswith("origin/"):
            default_branch = default_branch[len("origin/") :]

        return cls(
            cwd=str(cwd),
            repo_root=str(repo_root),
            branch=git(["branch", "--show-current"], "-") or "-",
            default_branch=default_branch,
            status=clip(git(["status", "--short"], "clean") or "clean", 1500),
            recent_commits=[line for line in git(["log", "--oneline", "-5"]).splitlines() if line],
            project_docs=docs,
        )

    def text(self) -> str:
        commits = "\n".join(f"- {line}" for line in self.recent_commits) or "- none"
        docs = "\n".join(f"- {path}\n{snippet}" for path, snippet in self.project_docs.items()) or "- none"
        return textwrap.dedent(
            f"""\
            Workspace:
            - cwd: {self.cwd}
            - repo_root: {self.repo_root}
            - branch: {self.branch}
            - default_branch: {self.default_branch}
            - status:
            {self.status}
            - recent_commits:
            {commits}
            - project_docs:
            {docs}
            """
        ).strip()

    def fingerprint(self) -> str:
        payload = {
            "cwd": self.cwd,
            "repo_root": self.repo_root,
            "branch": self.branch,
            "default_branch": self.default_branch,
            "status": self.status,
            "recent_commits": self.recent_commits,
            "project_docs": self.project_docs,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

