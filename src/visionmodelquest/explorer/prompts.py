from __future__ import annotations

import difflib
from dataclasses import dataclass

from visionmodelquest.contracts import SCENE_JSON_INSTRUCTION, SYSTEM_SAFETY

FREE_TEXT_INSTRUCTION = "Answer in plain text using no more than three concise sentences."


@dataclass(frozen=True)
class CompiledPrompt:
    system_instruction: str
    user_content: str
    response_contract_instruction: str

    def messages(self) -> tuple[dict[str, str], ...]:
        return (
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": self.user_content},
        )


def contract_instruction(contract: str) -> str:
    if contract == "scene_json_v1":
        return SCENE_JSON_INSTRUCTION
    if contract == "free_text_v1":
        return FREE_TEXT_INSTRUCTION
    raise ValueError(f"unknown contract: {contract}")


def compile_prompt(system_instruction: str, question: str, contract: str) -> CompiledPrompt:
    system = system_instruction.strip()
    user_question = question.strip()
    if not system:
        raise ValueError("system instruction must not be empty")
    if not user_question:
        raise ValueError("user question must not be empty")
    instruction = contract_instruction(contract)
    return CompiledPrompt(
        system_instruction=system,
        user_content=f"{user_question}\n\n{instruction}",
        response_contract_instruction=instruction,
    )


def is_canonical(system_instruction: str) -> bool:
    return system_instruction.strip() == SYSTEM_SAFETY


def canonical_diff(system_instruction: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            SYSTEM_SAFETY.splitlines(),
            system_instruction.strip().splitlines(),
            fromfile="canonical",
            tofile="experiment",
            lineterm="",
        )
    )
