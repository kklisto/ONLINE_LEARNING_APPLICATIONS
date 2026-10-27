# Online learning for pricing and advertising auctions

Academic team project studying sequential decisions under uncertainty: choosing prices, adapting to changing demand and bidding for advertising opportunities within a budget.

The repository contains simulation notebooks, reusable Python classes and the original project report. It is an experimental study rather than a deployed pricing or advertising service.

## Start here

Read the [team report](OLE%20project%20_%20Ameen%20and%20Kalisto.pdf), then open [requirement_1.ipynb](requirement_1.ipynb) for the stationary pricing and second-price auction setup. The report identifies Ameen and Kalisto; component-level individual ownership is not specified here.

## Experiments

| Notebook | Main questions |
| --- | --- |
| [Stationary pricing and second-price auctions](requirement_1.ipynb) | GP-UCB pricing, multiplicative pacing, UCB-style bidding and combined simulations |
| [First-price auctions](requirement_2.ipynb) | Non-truthful bidding, FF multiplicative pacing, UCB1 pricing and combined environments |
| [Non-stationary demand](requirement_3_extension1.ipynb) | UCB1, sliding-window UCB and CUSUM-UCB under changing demand |
| [Complementary products](requirement_3_extension2.ipynb) | Two-dimensional pricing with UCB and GP-UCB |
| [Competing bidders](requirement_4.ipynb) | How agent performance changes with budget, auction horizon and competition |

Supporting classes live in [importations/](importations/). The original [assignment](project.pdf) provides the task context.

## What to look for

- Exploration versus exploitation in pricing and bidding.
- Cumulative regret relative to the experiment's reference policy.
- Budget exhaustion and its effect on auction outcomes.
- The impact of abrupt demand changes on stationary algorithms.
- Sensitivity to arm discretisation, horizon and the number of advertising slots.

The notebooks illustrate why there is no universally best bidding strategy: results depend on the simulated market and resource constraints. Figures and observations are tied to those configurations, rather than claimed as a general performance guarantee.

## Run the experiments

The main simulations generate their environments in code and do not require an external dataset.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib jupyterlab
jupyter lab
```

Open notebooks from the repository root so that `from importations...` imports resolve, and run their cells in order. Review horizon, trial-count and discretisation settings before launching long simulations. The original runtime versions were not locked; the commands above describe the required packages rather than a validated historical environment.

## Limitations

Saved notebook plots are historical outputs. Random seeds and experimental assumptions should be checked for each comparison. Simulated demand, competitor bids and oracle/reference policies simplify real markets; no real revenue lift, advertiser deployment or production throughput is claimed.
