from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from feedback_agent.agent import FeedbackLoopAgent
from feedback_agent.config import load_config
from feedback_agent.llm import MockClient


def write_config(root: Path, workspace: Path, title: str, prompt: str) -> Path:
    config_path = root / "config.json"
    config_path.write_text(json.dumps({
        "implementation_model": {
            "name": "mock",
            "base_url": "http://127.0.0.1:1/v1",
            "api_key": "not-needed",
            "model": "local-gguf",
            "context_window": 4096,
            "max_tokens": 512,
            "temperature": 0.1,
        },
        "feedback_model": None,
        "mcp_tools": {"terminal": True, "web_scraping": True, "web_interaction": True},
        "runtime": {
            "docker_isolation": False,
            "docker_image": "feedback-loop-agent:local",
            "workspace": str(workspace),
            "command_timeout_seconds": 30,
        },
        "context_compaction": {
            "enabled": True,
            "threshold_ratio": 0.8,
            "keep_recent_turns": 4,
            "summary_max_tokens": 256,
        },
        "loop": {"max_iterations": 3, "stop_when_review_clean": True},
        "phases": {
            "requirements_refinement": {"max_iterations": 2},
            "plan_validation": {"max_iterations": 2},
            "implementation": {"max_iterations": 3},
        },
        "resolution_policy": {
            "max_same_error_repeats": 2,
            "allow_requirement_dilution": True,
            "allow_skip_with_note": True,
            "stop_on_cannot_resolve": False,
        },
        "project_design": {"title": title, "prompt": prompt},
    }), encoding="utf-8")
    return config_path


class FeedbackLoopAgentTests(unittest.TestCase):
    def test_mock_feedback_loop_has_distinct_phases_and_per_step_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            config_path = write_config(root, workspace, "mock tracker", "Build and test a task tracker.")
            cfg = load_config(config_path, repo_root=root)
            summary = FeedbackLoopAgent(cfg, mock=True).run()

            self.assertEqual(summary["requirements_refinement"]["status"], "resolved")
            self.assertEqual(summary["plan_validation"]["status"], "resolved")
            self.assertEqual(summary["final_status"], "resolved")
            self.assertGreaterEqual(len(summary["steps"]), 2)
            self.assertTrue(all(step["status"] == "resolved" for step in summary["steps"]))
            self.assertGreaterEqual(len(summary["steps"][0]["attempts"]), 2)
            self.assertTrue((workspace / "REQUIREMENTS.md").exists())
            self.assertTrue((workspace / "PLAN.md").exists())
            self.assertTrue((workspace / "task_tracker.py").exists())
            self.assertTrue((workspace / "test_task_tracker.py").exists())

    def test_mock_website_scenario_builds_browser_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "website"
            config_path = write_config(root, workspace, "mock website", "Build a tiny static website with a browser clicker game.")
            cfg = load_config(config_path, repo_root=root)
            summary = FeedbackLoopAgent(cfg, mock=True).run()

            self.assertEqual(summary["final_status"], "resolved")
            for name in ["index.html", "about.html", "game.html", "style.css", "app.js"]:
                self.assertTrue((workspace / name).exists(), name)
            self.assertIn("id=\"increment\"", (workspace / "game.html").read_text())
            self.assertIn("addEventListener", (workspace / "app.js").read_text())

    def test_mock_non_development_city_collection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "cities"
            config_path = write_config(root, workspace, "mock cities", "Use web scraping style workflow to collect images of big cities from Wikipedia.")
            cfg = load_config(config_path, repo_root=root)
            summary = FeedbackLoopAgent(cfg, mock=True).run()

            self.assertEqual(summary["final_status"], "resolved")
            manifest = json.loads((workspace / "city_image_manifest.json").read_text())
            self.assertGreaterEqual(len(manifest["cities"]), 4)
            self.assertTrue((workspace / "scripts" / "collect_city_images.py").exists())
            self.assertTrue((workspace / "collection_status.txt").exists())

    def test_platformer_review_rejects_failed_playwright_command(self) -> None:
        client = MockClient()
        prompt = json.dumps({
            "phase": "STEP_REVIEW_PHASE",
            "step": {"id": "S2", "title": "Add controllable movement and savepoints"},
            "requirements": {"project_summary": "platformer savepoint browser game"},
            "implementation": {
                "commands": [
                    {"command": ["python", "scripts/playwright_game_check.py"], "returncode": 1, "timed_out": False}
                ]
            },
        })

        review = json.loads(client.chat([{"role": "user", "content": prompt}]))

        self.assertEqual(review["status"], "needs_rework")
        self.assertIn("validation command failed", review["summary"])


if __name__ == "__main__":
    unittest.main()
