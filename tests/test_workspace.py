from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot import WorkspaceContext


class WorkspaceContextTests(unittest.TestCase):
    def test_repo_root_override_without_git_does_not_read_parent_repo_metadata(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="workspace-override-", dir=project_root) as temp:
            fixture = Path(temp) / "fixture"
            fixture.mkdir()
            (fixture / "README.md").write_text("fixture docs\n", encoding="utf-8")

            workspace = WorkspaceContext.build(fixture, repo_root_override=fixture)

            self.assertEqual(Path(workspace.repo_root), fixture.resolve())
            self.assertEqual(workspace.branch, "-")
            self.assertEqual(workspace.status, "clean")
            self.assertEqual(workspace.recent_commits, [])
            self.assertEqual(workspace.project_docs, {"README.md": "fixture docs\n"})

    def test_build_reads_git_metadata_for_real_repo(self):
        if shutil.which("git") is None:
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.run_git(root, "init")
            self.run_git(root, "config", "user.email", "minibot@example.test")
            self.run_git(root, "config", "user.name", "MiniBot Test")
            (root / "README.md").write_text("repo docs\n", encoding="utf-8")
            self.run_git(root, "add", "README.md")
            self.run_git(root, "commit", "-m", "init")

            workspace = WorkspaceContext.build(root)

            self.assertEqual(Path(workspace.repo_root), root.resolve())
            self.assertNotEqual(workspace.branch, "-")
            self.assertEqual(workspace.status, "clean")
            self.assertTrue(workspace.recent_commits)
            self.assertIn("README.md", workspace.project_docs)

    def run_git(self, cwd: Path, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
