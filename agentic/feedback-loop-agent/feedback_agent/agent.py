from __future__ import annotations

import json
from typing import Any

from .compaction import maybe_compact
from .config import AgentConfig
from .conversation import Conversation
from .llm import OpenAICompatClient, MockClient
from .workspace import (
    append_plan_note,
    collect_workspace_files,
    ensure_plan,
    extract_json_object,
    normalize_plan_steps,
    run_commands,
    write_files,
    write_plan_doc,
    write_requirements_doc,
)


REQUIREMENTS_CONTRACT = """
Return strict JSON only:
{
  "project_summary": "one paragraph",
  "refined_requirements": ["clear requirement"],
  "assumptions": ["explicit assumption or gap resolution"],
  "open_questions": [{"question": "gap", "resolution_strategy": "ask|assume|dilute|skip", "decision": "chosen resolution"}],
  "planning_confirmation": {
    "is_feasible": true,
    "is_clear": true,
    "is_verifiable": true,
    "verification_strategy": "how the plan will be checked step by step",
    "remaining_risks": ["risk or limitation"]
  },
  "plan": [
    {
      "id": "S1",
      "title": "short task title",
      "description": "what this task changes",
      "depends_on": [],
      "acceptance_criteria": ["verifiable criterion"],
      "validation_commands": [["python", "-m", "unittest", "-v"]]
    }
  ]
}
The plan must be ordered, distinct, and executable one step at a time.
"""


IMPLEMENTATION_CONTRACT = """
Return strict JSON only:
{
  "plan_note": "short note for PLAN.md",
  "files": [{"path": "relative/path", "content": "complete file content"}],
  "commands": [["python", "-m", "unittest", "-v"]],
  "resolution_request": "none|needs_requirements_change|needs_plan_change|cannot_resolve"
}
Only write paths inside the project workspace. Prefer small validation commands that finish quickly.
"""


REVIEW_STATUSES = {
    "resolved",
    "needs_rework",
    "cannot_resolve",
    "needs_requirements_change",
    "needs_plan_change",
    "skipped_with_note",
}


FEEDBACK_SYSTEM_PROMPT = """
You are the feedback/review agent in a two-agent development loop.
Read the full transcript, including implementation attempts, prior feedback,
requirements decisions, plan updates, command results, screenshots/reports when
listed, and unresolved risks. Be strict but bounded: push quality forward, ask
concrete cross-check questions, and decide whether the current phase is resolved,
needs rework, needs a plan/requirements change, or cannot be resolved.
Return strict JSON only.
"""


class FeedbackLoopAgent:
    """Orchestrates a phased two-agent workflow over one durable transcript.

    The implementation and feedback clients may point at the same local model or
    two different OpenAI-compatible endpoints. The orchestration layer owns the
    phase boundaries, file writes, validation commands, retry limits, and the
    transcript discipline that keeps long sessions coherent.
    """

    def __init__(self, config: AgentConfig, *, mock: bool = False):
        self.config = config
        self.workspace = config.runtime.workspace
        self.state_dir = self.workspace / ".agent_state"
        self.conversation = Conversation(self.state_dir / "conversation.jsonl")
        if mock:
            client = MockClient()
            self.impl_client = client
            self.feedback_client = client
        else:
            self.impl_client = OpenAICompatClient(config.implementation_model)
            self.feedback_client = OpenAICompatClient(config.feedback_model or config.implementation_model)
        self.requirements: dict[str, Any] = {}
        self.plan_steps: list[dict[str, Any]] = []
        self.plan_notes: list[str] = []

    def initialize(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        ensure_plan(self.workspace)
        if not self.conversation.turns:
            self.conversation.append(
                "system",
                (
                    "You are an agentic coding/workflow model. Work in explicit phases: "
                    "requirements refinement, plan validation, then one implementation feedback loop per plan step. "
                    "Maintain PLAN.md and REQUIREMENTS.md. Keep all work inside the project workspace. "
                    "This transcript is durable chat memory: IMPLEMENTATION_AGENT_REQUEST/RESPONSE and "
                    "FEEDBACK_AGENT_REQUEST/RESPONSE blocks are cumulative context, not isolated prompts."
                ),
            )
            self.conversation.append(
                "user",
                f"PROJECT DESIGN: {self.config.project_design.title}\n\n{self.config.project_design.prompt}",
            )

    def _implementation_chat(self, prompt: str) -> str:
        """Append a request, call the implementation model, then persist its reply.

        Keeping the request itself in the transcript matters. Without it, later
        turns see model answers but not the exact task/review that caused them,
        which makes long local-model sessions drift quickly.
        """
        maybe_compact(
            self.conversation,
            self.config,
            self.impl_client,
            context_window=self.config.implementation_model.context_window,
        )
        self.conversation.append("user", "IMPLEMENTATION_AGENT_REQUEST:\n" + prompt)
        raw = self.impl_client.chat(self.conversation.messages())
        self.conversation.append("assistant", "IMPLEMENTATION_AGENT_RESPONSE:\n" + raw)
        return raw

    def _feedback_chat(self, prompt: str, *, temperature: float = 0.1) -> str:
        """Run the feedback model against the same durable transcript.

        Feedback replies are stored as user-visible transcript blocks so the
        implementation model treats them as external critique on the next turn.
        The feedback model still receives the entire history, including its own
        previous reviews, which gives it continuity across loops.
        """
        feedback_cfg = self.config.feedback_model or self.config.implementation_model
        maybe_compact(
            self.conversation,
            self.config,
            self.feedback_client,
            context_window=feedback_cfg.context_window,
        )
        self.conversation.append("user", "FEEDBACK_AGENT_REQUEST:\n" + prompt)
        raw = self.feedback_client.chat(
            [
                {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
                *self.conversation.messages(system_as_user=True),
            ],
            temperature=temperature,
        )
        self.conversation.append("user", "FEEDBACK_AGENT_RESPONSE:\n" + raw)
        return raw

    def run(self) -> dict:
        self.initialize()
        req_result = self._requirements_refinement_phase()
        plan_result = self._plan_validation_phase()
        step_results: list[dict[str, Any]] = []
        for step in self.plan_steps:
            step_results.append(self._implementation_loop_for_step(step))
            write_plan_doc(self.workspace, self.requirements, self.plan_steps, self.plan_notes)
            if step_results[-1]["status"] == "cannot_resolve" and self.config.resolution_policy.stop_on_cannot_resolve:
                break
        summary = {
            "workspace": str(self.workspace),
            "requirements_refinement": req_result,
            "plan_validation": plan_result,
            "steps": step_results,
            "final_status": self._final_status(step_results),
        }
        (self.state_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _requirements_refinement_phase(self, extra_context: str | None = None) -> dict:
        """Turn an underspecified project brief into requirements and a draft plan."""
        iterations: list[dict[str, Any]] = []
        latest: dict[str, Any] = {}
        review: dict[str, Any] = {}
        for index in range(1, self.config.phases.requirements_refinement.max_iterations + 1):
            prompt = (
                f"REQUIREMENTS_REFINEMENT_PHASE iteration={index}\n"
                "Refine the project requirements before implementation. Fill gaps, record assumptions, "
                "and create a first ordered plan. Do not write project files yet. "
                "Before returning, answer the planning_confirmation fields: is the plan feasible, clear, "
                "and verifiable, and what exact verification strategy will later be enforced?\n"
                f"Extra context: {extra_context or 'none'}\n\n{REQUIREMENTS_CONTRACT}"
            )
            raw = self._implementation_chat(prompt)
            latest = extract_json_object(raw)
            self.requirements = latest
            self.plan_steps = normalize_plan_steps(latest.get("plan", []))
            for step in self.plan_steps:
                step.setdefault("status", "pending")
            write_requirements_doc(self.workspace, self.requirements)
            write_plan_doc(self.workspace, self.requirements, self.plan_steps, self.plan_notes)
            review = self._requirements_review(index, latest)
            iterations.append({"iteration": index, "requirements": latest, "review": review})
            if self._status(review) == "resolved":
                write_requirements_doc(self.workspace, self.requirements, review)
                append_plan_note(self.workspace, f"[requirements] resolved after iteration {index}: {review.get('summary', '')}")
                return {"status": "resolved", "iterations": iterations}
            self.conversation.append("user", "REQUIREMENTS_REWORK_DIRECTIVE:\nRevise requirements using this review:\n" + json.dumps(review, indent=2))
        fallback = self._fallback_resolution("requirements", review)
        self.requirements.setdefault("assumptions", []).append(fallback["note"])
        write_requirements_doc(self.workspace, self.requirements, review)
        return {"status": fallback["status"], "iterations": iterations, "resolution": fallback}

    def _requirements_review(self, index: int, requirements: dict[str, Any]) -> dict:
        """Ask the feedback agent whether requirements are actionable enough."""
        prompt = {
            "phase": "REQUIREMENTS_REVIEW_PHASE",
            "iteration": index,
            "project_design": self.config.project_design.prompt,
            "requirements": requirements,
            "expected_json": {
                "status": "resolved|needs_rework|needs_requirements_change|cannot_resolve|skipped_with_note",
                "needs_rework": True,
                "summary": "short review",
                "required_changes": ["specific change"],
                "cross_check_questions": ["requirement question the next pass must answer"],
            },
        }
        raw = self._feedback_chat(
            "REQUIREMENTS_REVIEW_PHASE\n"
            "Check whether the requirements are complete enough to support a distinct, verifiable plan. "
            "Reject vague requirements, missing gap decisions, and missing verification strategy.\n"
            + json.dumps(prompt),
            temperature=0.1,
        )
        review = extract_json_object(raw)
        return self._normalize_review(review)

    def _plan_validation_phase(self) -> dict:
        """Block implementation until the ordered plan is executable and checkable."""
        iterations: list[dict[str, Any]] = []
        review: dict[str, Any] = {}
        for index in range(1, self.config.phases.plan_validation.max_iterations + 1):
            review = self._plan_validation_review(index)
            iterations.append({"iteration": index, "review": review, "plan": self.plan_steps})
            if self._status(review) == "resolved":
                append_plan_note(self.workspace, f"[plan] validated after iteration {index}: {review.get('summary', '')}")
                write_plan_doc(self.workspace, self.requirements, self.plan_steps, self.plan_notes)
                return {"status": "resolved", "iterations": iterations}
            refined = self._plan_refinement_pass(index, review)
            iterations[-1]["refinement"] = refined
        fallback = self._fallback_resolution("plan", review)
        self.plan_notes.append(fallback["note"])
        write_plan_doc(self.workspace, self.requirements, self.plan_steps, self.plan_notes)
        return {"status": fallback["status"], "iterations": iterations, "resolution": fallback}

    def _plan_validation_review(self, index: int) -> dict:
        """Combine deterministic plan checks with model-based plan critique."""
        structural_findings = self._plan_structural_findings()
        prompt = {
            "phase": "PLAN_VALIDATION_PHASE",
            "iteration": index,
            "requirements": self.requirements,
            "plan": self.plan_steps,
            "deterministic_structural_findings": structural_findings,
            "checks": [
                "each step is distinct",
                "dependencies are explicit",
                "each step has acceptance criteria",
                "each step has validation commands or an explicit non-command validation method",
                "the sequence can be executed one step at a time",
                "planning_confirmation says the plan is feasible, clear, and verifiable",
                "the reviewer can name exactly how each step will be verified later",
            ],
            "expected_json": {
                "status": "resolved|needs_plan_change|needs_requirements_change|cannot_resolve",
                "needs_rework": True,
                "summary": "short review",
                "required_changes": ["specific change"],
                "planning_confirmation": {
                    "feasible": True,
                    "clear": True,
                    "verifiable": True,
                    "verification_matrix": [{"step_id": "S1", "how_verified": "command or explicit review method"}],
                },
            },
        }
        raw = self._feedback_chat(
            "PLAN_VALIDATION_PHASE\n"
            "Before implementation starts, explicitly confirm whether the plan is feasible, clear, "
            "and verifiable. If any step cannot be independently verified, return needs_plan_change.\n"
            + json.dumps(prompt),
            temperature=0.1,
        )
        return self._normalize_review(extract_json_object(raw))

    def _plan_refinement_pass(self, index: int, review: dict[str, Any]) -> dict:
        """Let the implementation model repair the plan while preserving context."""
        prompt = (
            f"PLAN_REFINEMENT_PHASE iteration={index}\n"
            "Revise only the ordered plan so every step is distinct, sequential, and verifiable. "
            "Keep requirements unless the review explicitly says they must change.\n"
            f"Current requirements: {json.dumps(self.requirements)}\n"
            f"Current plan: {json.dumps(self.plan_steps)}\n"
            f"Review: {json.dumps(review)}\n\n{REQUIREMENTS_CONTRACT}"
        )
        raw = self._implementation_chat(prompt)
        payload = extract_json_object(raw)
        if payload.get("refined_requirements"):
            self.requirements = payload
        self.plan_steps = normalize_plan_steps(payload.get("plan", self.plan_steps))
        self.plan_notes.append(f"Plan refined after review iteration {index}.")
        write_plan_doc(self.workspace, self.requirements, self.plan_steps, self.plan_notes)
        return payload

    def _implementation_loop_for_step(self, step: dict[str, Any]) -> dict:
        """Run bounded implement/review attempts for one validated plan step."""
        attempts: list[dict[str, Any]] = []
        same_error_count = 0
        last_summary = ""
        for attempt in range(1, self.config.phases.implementation.max_iterations + 1):
            maybe_compact(self.conversation, self.config, self.impl_client)
            implementation = self._implementation_pass(step, attempt)
            review = self._step_review_pass(step, attempt, implementation)
            attempts.append({"attempt": attempt, "implementation": implementation, "review": review})
            status = self._status(review)
            summary = str(review.get("summary", ""))
            same_error_count = same_error_count + 1 if summary == last_summary else 1
            last_summary = summary
            if status == "resolved":
                step["status"] = "resolved"
                append_plan_note(self.workspace, f"[{step['id']}] resolved: {summary}")
                return {"step_id": step["id"], "status": "resolved", "attempts": attempts}
            if status == "needs_plan_change":
                self._plan_refinement_pass(attempt, review)
            elif status == "needs_requirements_change":
                self._requirements_refinement_phase(extra_context=json.dumps(review))
            elif status == "cannot_resolve":
                step["status"] = "cannot_resolve"
                append_plan_note(self.workspace, f"[{step['id']}] cannot resolve: {summary}")
                return {"step_id": step["id"], "status": "cannot_resolve", "attempts": attempts}
            if same_error_count >= self.config.resolution_policy.max_same_error_repeats:
                resolution = self._fallback_resolution(f"step {step['id']}", review)
                step["status"] = resolution["status"]
                append_plan_note(self.workspace, f"[{step['id']}] {resolution['status']}: {resolution['note']}")
                return {"step_id": step["id"], "status": resolution["status"], "attempts": attempts, "resolution": resolution}
            self.conversation.append(
                "user",
                "NEXT_IMPLEMENTATION_DIRECTIVE:\nApply this step review in the next attempt. "
                "Keep previous requirements, plan validation, and this step context in mind:\n"
                + json.dumps(review, indent=2),
            )
        resolution = self._fallback_resolution(f"step {step['id']}", attempts[-1]["review"] if attempts else {})
        step["status"] = resolution["status"]
        return {"step_id": step["id"], "status": resolution["status"], "attempts": attempts, "resolution": resolution}

    def _implementation_pass(self, step: dict[str, Any], attempt: int) -> dict:
        """Ask for complete-file edits and run the model-requested validations."""
        prompt = (
            f"IMPLEMENT_PLAN_STEP_PHASE step_id={step['id']} attempt={attempt}\n"
            "Work on this single plan step only. Do not silently jump ahead. If the step is impossible, "
            "use resolution_request and explain why. Cross-check your edits against this step's acceptance "
            "criteria and include validation commands that prove the step whenever terminal tools are enabled.\n"
            f"Refined requirements: {json.dumps(self.requirements)}\n"
            f"Full validated plan: {json.dumps(self.plan_steps)}\n"
            f"Current step: {json.dumps(step)}\n\n{IMPLEMENTATION_CONTRACT}"
        )
        raw = self._implementation_chat(prompt)
        payload = extract_json_object(raw)
        written = write_files(self.workspace, payload.get("files", []))
        command_results = []
        if self.config.mcp_tools.terminal:
            command_results = run_commands(
                self.workspace,
                payload.get("commands", []),
                self.config.runtime.command_timeout_seconds,
            )
        note = payload.get("plan_note") or f"{step['id']} attempt {attempt} implementation pass completed."
        append_plan_note(self.workspace, f"[{step['id']} attempt {attempt}] {note}")
        return {"written": written, "commands": command_results, "raw": payload}

    def _step_review_pass(self, step: dict[str, Any], attempt: int, implementation: dict[str, Any]) -> dict:
        """Critique one step using files, command results, and prior transcript."""
        plan_text = (self.workspace / "PLAN.md").read_text(encoding="utf-8")
        prompt = {
            "phase": "STEP_REVIEW_PHASE",
            "step": step,
            "attempt": attempt,
            "requirements": self.requirements,
            "plan": self.plan_steps,
            "plan_file": plan_text,
            "implementation": implementation,
            "files": collect_workspace_files(self.workspace),
            "review_instructions": [
                "Run-result failures, timeouts, missing files, broken UI hooks, and weak validation must be called out.",
                "Ask concrete cross-check questions against the refined requirements and the current step acceptance criteria.",
                "Use command results and workspace files as evidence; do not accept a step just because files were written.",
                "For browser/game work, require Playwright-style interaction evidence and screenshot/report artifacts when configured.",
                "Return needs_plan_change if this step cannot be independently verified as written.",
                "Return needs_requirements_change if the requirements are contradictory or impossible.",
                "Return cannot_resolve only when bounded retries are unlikely to help.",
            ],
            "expected_json": {
                "status": "resolved|needs_rework|cannot_resolve|needs_requirements_change|needs_plan_change|skipped_with_note",
                "needs_rework": True,
                "summary": "short review",
                "required_changes": ["specific change"],
                "cross_check_questions": ["question answered by code/commands/files"],
                "verification_evidence": ["command/file/screenshot/report checked"],
            },
        }
        raw = self._feedback_chat(
            "STEP_REVIEW_PHASE\n"
            "Critically verify exactly one plan step. Use the whole transcript to avoid repeating old mistakes, "
            "but judge only the current step against its acceptance criteria.\n"
            + json.dumps(prompt),
            temperature=0.1,
        )
        review = self._normalize_review(extract_json_object(raw))
        append_plan_note(self.workspace, f"[{step['id']} attempt {attempt}] review: {review.get('summary', 'no summary')}")
        return review

    def _plan_structural_findings(self) -> list[str]:
        """Cheap deterministic guardrails before the model-based plan review.

        The reviewer still makes the judgment call, but these findings prevent
        obvious misses such as empty validation commands or broken dependencies
        from slipping through just because a model review was too generous.
        """
        findings: list[str] = []
        if not self.plan_steps:
            return ["Plan has no steps."]
        seen_ids: set[str] = set()
        for step in self.plan_steps:
            step_id = str(step.get("id") or "<missing>")
            if step_id in seen_ids:
                findings.append(f"Duplicate step id: {step_id}.")
            seen_ids.add(step_id)
            if not step.get("acceptance_criteria"):
                findings.append(f"{step_id} has no acceptance criteria.")
            if not step.get("validation_commands"):
                findings.append(f"{step_id} has no validation commands or explicit validation method.")
            for dep in step.get("depends_on", []):
                if dep not in seen_ids:
                    findings.append(f"{step_id} depends on {dep}, which has not appeared earlier in the ordered plan.")
        confirmation = self.requirements.get("planning_confirmation") if isinstance(self.requirements, dict) else None
        if not isinstance(confirmation, dict):
            findings.append("Requirements are missing planning_confirmation.")
        else:
            for key in ("is_feasible", "is_clear", "is_verifiable"):
                if confirmation.get(key) is not True:
                    findings.append(f"planning_confirmation.{key} is not true.")
            if not confirmation.get("verification_strategy"):
                findings.append("planning_confirmation.verification_strategy is empty.")
        return findings

    def _status(self, review: dict[str, Any]) -> str:
        status = str(review.get("status") or "").strip()
        if status in REVIEW_STATUSES:
            return status
        if review.get("needs_rework") is False:
            return "resolved"
        return "needs_rework"

    def _normalize_review(self, review: dict[str, Any]) -> dict[str, Any]:
        review = dict(review)
        status = self._status(review)
        review["status"] = status
        review["needs_rework"] = status not in {"resolved", "skipped_with_note"}
        review.setdefault("summary", "no summary")
        review.setdefault("required_changes", [])
        return review

    def _fallback_resolution(self, scope: str, review: dict[str, Any]) -> dict[str, str]:
        """Choose a bounded outcome when retries stop making progress."""
        summary = review.get("summary", "No final review summary.") if review else "No review was produced."
        if self.config.resolution_policy.allow_skip_with_note:
            status = "skipped_with_note"
            note = f"Bounded retries exhausted for {scope}; skipped with note. Last review: {summary}"
        elif self.config.resolution_policy.allow_requirement_dilution:
            status = "needs_requirements_change"
            note = f"Bounded retries exhausted for {scope}; requirements must be diluted or clarified. Last review: {summary}"
        else:
            status = "cannot_resolve"
            note = f"Bounded retries exhausted for {scope}; cannot resolve. Last review: {summary}"
        return {"status": status, "note": note}

    def _final_status(self, step_results: list[dict[str, Any]]) -> str:
        if not step_results:
            return "no_steps"
        statuses = {item["status"] for item in step_results}
        if statuses == {"resolved"}:
            return "resolved"
        if "cannot_resolve" in statuses:
            return "cannot_resolve"
        if "skipped_with_note" in statuses:
            return "resolved_with_skips"
        return "partial"
