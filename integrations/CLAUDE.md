# Claude Code Integration: Evren-Code Subagent

This project and workspace supports **Evren-Code** as an autonomous subagent powered by the **EVREN AI Platform** models (`DeepSeek-V4.1 Flash`, `GLM-5.3`, `Qwen-3.8 Flash Next`, `Gemma-4 31B`, etc.).

---

## 🛠️ Claude Code Subagent Delegation Rules

When complex architectural planning, surgical code refactoring, or EVREN-specific models are required, invoke `evren-code` as a specialized subagent using either of the following approaches:

### Method 1: Headless CLI & Structured JSON Mode (Recommended)

Execute via Claude Code terminal / bash tool:

```bash
# 1. General Autonomous Coding or Problem Resolution:
evren --headless --json "Implement the requested feature"

# 2. Architectural Design & Risk Analysis with DeepSeek V4.1:
evren --headless --json --role architect "Design the authentication module architecture"

# 3. High-Speed Surgical Coding with Qwen 3.8 Flash Next:
evren --headless --json --role coder "Add JWT refresh token rotation logic"

# 4. Rigorous Code Review:
evren --headless --json --role reviewer "Review src/auth.py for security and edge cases"

# 5. Direct Raw LLM Completion (Without File Edits / Turn Loop):
# Deep reasoning (thinking) via DeepSeek V4.1 Flash:
evren ask --json -m deepseek-v4.1-flash "Analyze the query execution plan bottleneck: SELECT ..."

# Raw code generation via Qwen 3.8 Flash Next:
evren ask --json -m qwen3.8-flash-next "Write a FastAPI rate-limiting dependency function"
```

The output is structured JSON:
```json
{
  "status": "success",
  "model": "deepseek-v4.1-flash",
  "thinking": "Model reasoning steps...",
  "content": "Final model response...",
  "usage": { "input": 45, "output": 180 }
}
```

---

### Method 2: Model Context Protocol (MCP)

When the MCP server is configured (`evren mcp`), call native tools directly:
* `evren_llm_call(prompt, model, system_prompt)`: **Direct raw LLM call.** Queries DeepSeek-V4.1, Qwen-3.8, GLM-5.3, or Gemma-4 and returns reasoning/response without modifying files.
* `evren_task(task, workspace, role)`: Autonomous end-to-end coding and refactoring tasks.
* `evren_architect(task, workspace)`: Deep architectural design and risk assessment.
* `evren_code_review(path, workspace)`: Security, code quality, and maintainability review.
* `evren_models()`: Catalog active models, roles, and context limits.
