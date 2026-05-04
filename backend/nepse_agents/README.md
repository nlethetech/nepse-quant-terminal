# nepse_agents

Clean public multi-agent workflow for NEPSE Quant Terminal.

This package keeps the architecture of the private trading-agent system while
removing private research data, private strategy names, credentials, execution
integrations, and live execution. It is intentionally paper-only.

Pipeline:

```text
Evidence inputs
-> Quant Strategy Engine
-> Research Team
-> Bull/Bear Debate
-> Trader
-> Risk Committee
-> Portfolio Manager
-> Paper Decision
-> Checkpoint
```

Public guarantees:

- No live order path.
- No credential or session handling.
- No proprietary strategy parameters.
- Evidence quorum before approval.
- Checkpoint written for every run.

Example:

```python
from backend.nepse_agents import run_paper_decision

result = run_paper_decision({
    "symbol": "NABIL",
    "score": 0.76,
    "confidence": 0.70,
    "suggested_quantity": 20,
    "suggested_limit_price": 500,
    "evidence_refs": ["public-signal:demo"],
})
print(result["portfolio_decision"])
```
