# rsn-game-qa

Grounded game QA experiments: YOLO + RL + LLM agents that play games from pixels, explore bugs and balance issues, and generate human‑readable test reports.

## Current status (2026-02-08)

- Repo renamed and cleaned (`rsn-game-qa`).
- Basic project structure in place:

```bash
├───data
│   ├───live_logs
│   │   ├───backup
│   │   ├───run_20260208T140712_rl_help_shadowrun
│   │   └───run_20260208T190624_breakout_71_v1
│   └───twin_dataset
│       ├───images
│       └───labels
├───documentation
│   ├───BigRocks
│   └───business
├───runs
│   └───rl_help_v3
├───scripts
└───src
    ├───agents
    ├───controllers
    ├───env
    ├───perception
    ├───policies
    └───rl
```

- Next step: implement a minimal Breakout‑71 controller and perception stub
  that can move the paddle under program control.

## How to run

For Breakout 71, the paddle is controlled by **press-and-hold plus random horizontal movement**.
The controller simulates this by using long swipes at the bottom of the screen:

- It “presses” near the paddle row,
- Slides left or right while still pressed for a period of time
- Then stops

```bash
python src/agents/run_breakout_controller.py
```
