"""
inference.py — Baseline agent for Incident Response Commander
Uses OpenAI-compatible client via LiteLLM proxy.

Required environment variables:
    API_BASE_URL   → LiteLLM proxy base URL
    MODEL_NAME     → model identifier (e.g. meta-llama/Llama-3.3-70B-Instruct)
    HF_TOKEN       → Hugging Face / proxy API key
"""

import os
import json
import re
from openai import OpenAI
from env.environment import IncidentResponseEnv
from env.models import Action

# ── Config ────────────────────────────────────────────────────
MAX_STEPS   = 10
TEMPERATURE = 0.2

# ── OpenAI client (initialised once) ─────────────────────────
def get_client() -> OpenAI:
    api_base_url = os.environ.get("API_BASE_URL", "").rstrip("/")
    hf_token     = os.environ.get("HF_TOKEN", "")

    if not api_base_url:
        raise RuntimeError("API_BASE_URL environment variable is not set")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable is not set")

    return OpenAI(base_url=api_base_url, api_key=hf_token)


# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert on-call engineer responding to a production incident.
Your job is to investigate alerts, find the root cause, fix it, and notify your team.

You can take ONE action per turn. Choose from:
- check_logs      target=<service_name>
- check_metrics   target=<service_name>
- restart_service target=<service_name>
- rollback_deploy target=<service_name>
- scale_up        target=<service_name>
- escalate        target=<reason>
- notify_team     message=<your message>

Rules:
1. Always investigate (check_logs or check_metrics) BEFORE taking a fix action
2. Only fix the service that is actually broken
3. Do NOT end without calling notify_team — it is MANDATORY
4. Call notify_team BEFORE applying the fix
5. Do not restart or rollback services that are healthy

IMPORTANT: Your sequence should always be:
  investigate → notify_team → fix

Respond with ONLY a JSON object like this:
{"action": "check_logs", "target": "payment-service"}
or
{"action": "notify_team", "message": "payment-service is down, fixing now"}

Nothing else. No explanation. Just the JSON.
""".strip()


# ── Build prompt from observation ─────────────────────────────
def build_prompt(obs, step_history):
    services_str = "\n".join([
        f"  - {s.name}: status={s.status}, cpu={s.cpu}%, memory={s.memory}%, error_rate={s.error_rate}"
        for s in obs.services
    ])
    alerts_str = "\n".join([
        f"  - [{a.severity.upper()}] {a.service}: {a.message}"
        for a in obs.alerts
    ])
    history_str = "\n".join(step_history[-5:]) if step_history else "None"

    return f"""
CURRENT INCIDENT — Step {obs.step}
Actions remaining: {obs.actions_remaining}

ACTIVE ALERTS:
{alerts_str}

SERVICE STATUS:
{services_str}

LAST ACTION RESULT:
{obs.last_action_result}

WHAT YOU HAVE DONE SO FAR:
{history_str}

What is your next action? Respond with JSON only.
""".strip()


# ── Parse LLM response into Action ───────────────────────────
def parse_action(response_text: str) -> Action:
    try:
        match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return Action(
                name=data.get("action", "escalate"),
                target=data.get("target"),
                message=data.get("message"),
            )
    except Exception:
        pass
    print(f"  ⚠️  Could not parse response: {response_text[:100]}", flush=True)
    return Action(name="escalate", target="parse_error")


# ── Run one episode ───────────────────────────────────────────
def run_episode(task: str) -> dict:
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

    print(f"\n{'='*60}", flush=True)
    print(f"TASK: {task.upper()}", flush=True)
    print('='*60, flush=True)
    print(f"[START] task={task}", flush=True)

    env          = IncidentResponseEnv(task=task, max_steps=MAX_STEPS)
    obs          = env.reset()
    step_history = []
    grade_result = None
    client       = get_client()

    print(f"Incident: {env._scenario['description']}", flush=True)
    print(f"Alerts  : {[a.message for a in obs.alerts]}", flush=True)

    for step in range(1, MAX_STEPS + 1):
        prompt = build_prompt(obs, step_history)

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=100,
            )
            response_text = completion.choices[0].message.content or ""
            print(f"[DEBUG] API call succeeded", flush=True)
        except Exception as e:
            import traceback
            print(f"  ❌ LLM call HARD FAILED: {e}", flush=True)
            traceback.print_exc()
            raise

        action = parse_action(response_text)
        print(f"\n  Step {step}: LLM chose → {action.name}({action.target or action.message or ''})", flush=True)

        obs, reward, done, info = env.step(action)
        print(f"  Result  : {obs.last_action_result}", flush=True)
        print(f"  Reward  : {reward.value} ({reward.reason})", flush=True)
        print(f"[STEP] task={task} step={step} reward={reward.value} done={done}", flush=True)

        step_history.append(
            f"Step {step}: {action.name}({action.target or action.message or ''}) "
            f"→ reward {reward.value}"
        )

        if done:
            grade_result = info.get("grade")
            break

    if not grade_result:
        from env.grader import IncidentGrader
        grader       = IncidentGrader(env._scenario)
        grade_result = grader.grade(
            action_history=env._action_history,
            resolved=env._scenario.get("resolved", False)
        )

    print(f"[END] task={task} score={grade_result['score']} steps={len(env._action_history)} passed={int(grade_result['passed'])}", flush=True)
    print(f"\n── FINAL GRADE ────────────────────────────────", flush=True)
    print(f"  Score   : {grade_result['score']} / 1.0", flush=True)
    print(f"  Passed  : {'✅ YES' if grade_result['passed'] else '❌ NO'}", flush=True)
    print(f"  Feedback:", flush=True)
    for f in grade_result["feedback"]:
        print(f"    {f}", flush=True)

    return grade_result


# ── Main ──────────────────────────────────────────────────────
def main():
    api_base_url = os.environ.get("API_BASE_URL", "")
    hf_token     = os.environ.get("HF_TOKEN", "")
    model_name   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

    print("🚨 Incident Response Commander — Baseline Agent", flush=True)
    print(f"   Model : {model_name}", flush=True)
    print(f"   API   : {api_base_url}", flush=True)
    print(f"[DEBUG] HF_TOKEN present: {bool(hf_token)}", flush=True)
    print(f"[DEBUG] MODEL_NAME: {model_name}", flush=True)

    results = {}
    for task in ["easy", "medium", "hard"]:
        results[task] = run_episode(task)

    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print('='*60, flush=True)
    total = 0
    for task, result in results.items():
        score  = result["score"]
        passed = "✅" if result["passed"] else "❌"
        print(f"  {task:<10} {passed}  {score} / 1.0", flush=True)
        total += score

    avg = round(total / len(results), 3)
    print(f"\n  Average score: {avg} / 1.0", flush=True)
    print(f"  Tasks passed : {sum(1 for r in results.values() if r['passed'])} / {len(results)}", flush=True)


if __name__ == "__main__":
    main()
