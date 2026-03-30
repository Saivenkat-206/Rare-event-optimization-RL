# Reinforcement Learning for Power Grid Maintenance under Catastrophic Risk

## Overview

This project explores a key limitation of standard Reinforcement Learning (RL): **tail-risk blindness**.

We simulate a simplified power grid where an agent must decide whether to:

* Perform maintenance (incurs cost)
* Skip maintenance (saves cost but increases risk)

The challenge is that **failures are rare but extremely costly**.
A naïve RL agent optimizing expected reward often ignores these rare events, leading to catastrophic outcomes.

This project demonstrates:

* Failure of naïve DQN under rare-event risk
* Improvement using a simple risk-aware modification
* Clear behavioral differences via visualization

---

## Problem Intuition

Real-world systems (power grids, aviation, finance) face:

* Gradual degradation over time
* Limited maintenance budgets
* Rare but high-impact failures

This creates a tradeoff:

> Spend now to stay safe
> vs
> Save now and risk catastrophic failure later

---

## Environment Design

The environment is a stylized abstraction (not a real grid simulator).

### 🔹 State Space

* **Health (Preparedness)**: Stability of the system (0 → 1)
* **Time Since Maintenance**
* **Budget (Resources)**

### 🔹 Actions

* `0` → Do nothing
* `1` → Perform maintenance

### 🔹 Dynamics

* Health **decays over time**
* Maintenance **improves health but costs budget**
* Budget is **limited and continuously depleted**

### 🔹 Failures

Failures are **probabilistic**, not deterministic:

* Lower health → higher probability of failure
* Failures cause **large negative rewards**
* Even high-health systems can fail (low probability)

---

## Key Insight: Tail Risk

Standard RL optimizes **expected reward**, which leads to:

* Ignoring rare catastrophic events
* Delaying maintenance
* Operating in high-risk states
* Sudden collapse when failure occurs

---

## Agents Compared

### Naïve DQN

* Optimizes expected reward
* Reactive behavior
* Delays maintenance
* High variance and frequent failures

---

### Risk-Aware DQN

* Adds **continuous risk penalty**
* Encourages proactive maintenance
* Reduces time spent in dangerous states
* Fewer catastrophic failures

---

## Results

| Metric          | Naïve Agent | Risk-Aware Agent |
| --------------- | ----------- | ---------------- |
| Total Reward    | Worse       | Better           |
| Failures        | More        | Fewer            |
| Maintenance Use | Lower       | Higher           |
| Stability       | Poor        | Improved         |

### Observations

* Naïve agent collapses early due to neglect
* Risk-aware agent spends more but survives longer
* Tradeoff: **cost vs safety**

---

## Visualization (Streamlit UI)

The project includes an interactive UI to compare agents:

* Preparedness (Health) over time
* Budget usage
* Failure events (marked)
* Total reward and statistics

Run:

```bash
streamlit run app.py
```

---

## How to Run

### 1. Install dependencies

```bash
pip install torch numpy matplotlib streamlit gymnasium
```

---

### 2. Train agents

```bash
python NaiveDQNTrain.py
python FixedDQNTrain.py
```

---

### 3. Launch UI

```bash
streamlit run app.py
```

---

## Real-World Applications

This framework maps to several domains:

* Power grid maintenance
* Aircraft inspection scheduling
* Industrial equipment reliability
* Financial risk management
* Climate infrastructure planning

---

## Key Takeaway

> Optimizing for average outcomes is not enough in systems with rare catastrophic risks.

Risk-aware strategies:

* Reduce extreme failures
* Improve long-term stability
* Accept higher short-term cost for safety

---

