from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    context_window: int
    max_tokens: int
    temperature: float


@dataclass(frozen=True)
class ToolConfig:
    terminal: bool
    web_scraping: bool
    web_interaction: bool


@dataclass(frozen=True)
class RuntimeConfig:
    docker_isolation: bool
    docker_image: str
    workspace: Path
    command_timeout_seconds: int


@dataclass(frozen=True)
class CompactionConfig:
    enabled: bool
    threshold_ratio: float
    keep_recent_turns: int
    summary_max_tokens: int


@dataclass(frozen=True)
class LoopConfig:
    max_iterations: int
    stop_when_review_clean: bool


@dataclass(frozen=True)
class PhaseLoopConfig:
    max_iterations: int


@dataclass(frozen=True)
class PhaseConfig:
    requirements_refinement: PhaseLoopConfig
    plan_validation: PhaseLoopConfig
    implementation: PhaseLoopConfig


@dataclass(frozen=True)
class ResolutionPolicy:
    max_same_error_repeats: int
    allow_requirement_dilution: bool
    allow_skip_with_note: bool
    stop_on_cannot_resolve: bool


@dataclass(frozen=True)
class ProjectDesign:
    title: str
    prompt: str


@dataclass(frozen=True)
class AgentConfig:
    implementation_model: ModelConfig
    feedback_model: ModelConfig | None
    mcp_tools: ToolConfig
    runtime: RuntimeConfig
    context_compaction: CompactionConfig
    loop: LoopConfig
    phases: PhaseConfig
    resolution_policy: ResolutionPolicy
    project_design: ProjectDesign


def _model(data: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        name=str(data["name"]),
        base_url=str(data["base_url"]).rstrip("/"),
        api_key=str(data.get("api_key") or "not-needed"),
        model=str(data.get("model") or "local-gguf"),
        context_window=int(data["context_window"]),
        max_tokens=int(data.get("max_tokens", 2048)),
        temperature=float(data.get("temperature", 0.25)),
    )


def _phase_loop(data: dict[str, Any], key: str, default: int) -> PhaseLoopConfig:
    value = data.get(key, {})
    return PhaseLoopConfig(max_iterations=int(value.get("max_iterations", default)))


def _phases(data: dict[str, Any], loop_data: dict[str, Any]) -> PhaseConfig:
    phase_data = data.get("phases", {})
    old_loop_iterations = int(loop_data.get("max_iterations", 3))
    return PhaseConfig(
        requirements_refinement=_phase_loop(phase_data, "requirements_refinement", 2),
        plan_validation=_phase_loop(phase_data, "plan_validation", 2),
        implementation=_phase_loop(phase_data, "implementation", old_loop_iterations),
    )


def _resolution_policy(data: dict[str, Any]) -> ResolutionPolicy:
    policy = data.get("resolution_policy", {})
    return ResolutionPolicy(
        max_same_error_repeats=int(policy.get("max_same_error_repeats", 2)),
        allow_requirement_dilution=bool(policy.get("allow_requirement_dilution", True)),
        allow_skip_with_note=bool(policy.get("allow_skip_with_note", True)),
        stop_on_cannot_resolve=bool(policy.get("stop_on_cannot_resolve", False)),
    )


def load_config(path: str | Path, repo_root: Path | None = None) -> AgentConfig:
    cfg_path = Path(path).resolve()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = repo_root or cfg_path.parents[2]

    feedback = data.get("feedback_model")
    runtime_data = data["runtime"]
    workspace_override = os.getenv("AGENT_WORKSPACE")
    workspace = Path(workspace_override or runtime_data["workspace"])
    if not workspace.is_absolute():
        workspace = (base / workspace).resolve()

    loop_data = data.get("loop", {})

    return AgentConfig(
        implementation_model=_model(data["implementation_model"]),
        feedback_model=_model(feedback) if feedback else None,
        mcp_tools=ToolConfig(**data["mcp_tools"]),
        runtime=RuntimeConfig(
            docker_isolation=bool(runtime_data.get("docker_isolation", True)),
            docker_image=str(runtime_data.get("docker_image", "feedback-loop-agent:local")),
            workspace=workspace,
            command_timeout_seconds=int(runtime_data.get("command_timeout_seconds", 120)),
        ),
        context_compaction=CompactionConfig(**data["context_compaction"]),
        loop=LoopConfig(
            max_iterations=int(loop_data.get("max_iterations", 3)),
            stop_when_review_clean=bool(loop_data.get("stop_when_review_clean", True)),
        ),
        phases=_phases(data, loop_data),
        resolution_policy=_resolution_policy(data),
        project_design=ProjectDesign(**data["project_design"]),
    )
