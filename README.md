# 🏭 Supply Chain Agility Simulator

Interactive simulation tool demonstrating the impact of **supply chain agility** on service levels and financial performance in the luxury industry.

## What it does

Visualizes a complete supply chain (Supplier → Raw Material → Semi-Finished → Finished Product → Warehouse → Store) with:

- **Week-by-week animation** of goods flowing through the pipeline
- **Real-time KPIs** (service level, revenue, margin) updating as you scrub through time
- **Configurable parameters**: lead times, order frequency, initial stock levels, demand profiles, capacity constraints, economics
- **Scenario comparison**: save scenarios and compare agile (weekly ordering) vs non-agile (monthly ordering) strategies

## Key insight

When demand surges unexpectedly, an **agile supply chain** (weekly ordering, distributed WIP) recovers 3+ weeks faster than a traditional push model — translating to hundreds of thousands in recovered margin.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `app.py` → Deploy

## Scenarios to try

| Scenario | Order Freq | Init Store | Init CW | Init Semi | Init RawMat |
|----------|-----------|------------|---------|-----------|-------------|
| SC1: Non-Agile Push | 4 weeks | 1500 | 0 | 0 | 0 |
| SC2: Agile Weekly | 1 week | 1200 | 0 | 0 | 0 |
| SC3: Agile + Distributed WIP | 1 week | 600 | 200 | 200 | 400 |

## Supply chain logic

- **Forecast**: Naïve (= this week's observed demand), updated each Friday
- **Ordering**: Friday evening — see demand → update forecast → order to cover (LT + frequency) weeks
- **Capacity**: Each production stage starts at 100 pcs/wk, ramps up max +20%/wk when running at capacity
- **Flow conservation**: Every unit is tracked — demand = sales + missed sales (verified)

## License

MIT
