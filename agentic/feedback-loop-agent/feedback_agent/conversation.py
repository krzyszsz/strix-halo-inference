from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class Turn:
    role: str
    content: str


class Conversation:
    def __init__(self, path: Path):
        self.path = path
        self.turns: list[Turn] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    self.turns.append(Turn(role=item["role"], content=item["content"]))

    def append(self, role: str, content: str) -> None:
        turn = Turn(role=role, content=content)
        self.turns.append(turn)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")

    def messages(self) -> list[dict[str, str]]:
        return [asdict(t) for t in self.turns]

    def estimated_tokens(self) -> int:
        return max(1, sum(len(t.content) for t in self.turns) // 4)

    def replace_with_memory(self, memory: str, keep_recent_turns: int) -> None:
        recent = self.turns[-keep_recent_turns:] if keep_recent_turns > 0 else []
        self.turns = [
            Turn(
                role="system",
                content=(
                    "Compacted durable memory from earlier turns. Preserve these decisions, "
                    "constraints, and unresolved risks:\n\n" + memory
                ),
            ),
            *recent,
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "".join(json.dumps(asdict(t), ensure_ascii=False) + "\n" for t in self.turns),
            encoding="utf-8",
        )
