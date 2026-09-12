# Computational Cost

Statistics cover **5 viewpoints** from the latest run.

| Major module | Invocations | Calls/viewpoint | Latency/call (s) | Amortized s/viewpoint | GPT calls | GPT latency/call (s) | Prompt tokens/call | Completion tokens/call | Total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bev_build | 5 | 1.000 | 24.086 | 24.086 | 0 | 0.000 | 0.0 | 0.0 | 0 |
| dcll_skg_update | 5 | 1.000 | 3.555 | 3.555 | 5 | 3.425 | 1656.0 | 216.4 | 9362 |
| synchronize | 3 | 0.600 | 8.333 | 5.000 | 2 | 12.497 | 472.0 | 53.0 | 1050 |
| align | 5 | 1.000 | 12.691 | 12.691 | 11 | 5.726 | 1436.4 | 234.8 | 18383 |
| backtrack | 0 | 0.000 | 0.000 | 0.000 | 0 | 0.000 | 0.0 | 0.0 | 0 |

Reviewer response:

We report computational cost at the viewpoint level for the five major modules: BEV construction, DCLL/SKG update, synchronization, alignment, and backtracking. Conditional latency is measured only when a module is invoked, while amortized latency divides its total wall-clock time by all 5 evaluated viewpoints. Across this run, these modules used 226.662 seconds in total, including 18 GPT API calls and 28795 tokens. The table reports module-level API latency, call count, and prompt/completion token usage.
