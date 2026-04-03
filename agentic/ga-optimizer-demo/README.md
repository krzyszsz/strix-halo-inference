# GA Optimizer Demo (Agentic, Staged)

This demo implements a multithreaded genetic algorithm optimizer for `double[]` in C# with a staged workflow:

1. Interface and contracts first.
2. Tests written before implementation.
3. Empty implementation intentionally failing.
4. MCP benchmark preparation step.
5. Full implementation and passing tests.

## Project Layout

- `src/GeneticOptimizer/` — optimizer contracts + implementation.
- `tests/GeneticOptimizer.Tests/` — behavior tests.
- `out/` — logs and MCP evidence artifacts.

## Run

```bash
export REPO_ROOT="$(pwd)"
source "$REPO_ROOT/scripts/env.sh"
cd "$REPO_ROOT"

python agentic/ga-optimizer-demo/scripts/mcp_prepare_benchmark.py
dotnet test agentic/ga-optimizer-demo/GaOptimizerDemo.slnx
```

Or:

```bash
bash agentic/ga-optimizer-demo/scripts/run_tests.sh
```

## Evidence

- Publish-day rerun log:
  - `reports/publish/agentic_ga_optimizer_tests.log`
- Local test output:
  - `agentic/ga-optimizer-demo/out/tests_post_impl_recheck_publish.log`
