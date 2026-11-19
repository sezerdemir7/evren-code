---
name: evren-code
description: Specialized software engineering subagent that utilizes EVREN AI platform models (DeepSeek V4.1, GLM 5.3, Qwen 3.8, Gemma 4) for autonomous coding, architecture planning, and rigorous code reviews.
---

# Evren-Code Antigravity Skill

Use this skill when you need to leverage EVREN AI platform models, execute autonomous software engineering tasks on the EVREN infrastructure, or delegate deep architecture and code review tasks.

## Usage Methods

### 1. Headless Invocation via `run_command` (JSON Output)

```powershell
evren --headless --json -C "<workspace_dir>" "<task>"
```

Targeting a specialized role:
* **Architecture & Analysis (`deepseek-v4.1-flash`):**
  `evren --headless --json --role architect -C "<workspace_dir>" "<task>"`
* **Surgical Fast Coding (`qwen3.8-flash-next`):**
  `evren --headless --json --role coder -C "<workspace_dir>" "<task>"`
* **Code Review & Quality (`deepseek-v4.1-flash`):**
  `evren --headless --json --role reviewer -C "<workspace_dir>" "<task>"`
* **Direct Raw LLM Queries (Without File Modifications):**
  `evren ask --json -m deepseek-v4.1-flash "<query>"`
  `evren ask --json -m qwen3.8-flash-next "<code completion query>"`

### 2. Standard JSON Output

The resulting JSON includes `status`, `summary`, `modified_files`, and `model_used` properties for structured consumption.
