# NCAA Baseball Predictive Models

Bayesian player projection models for NCAA Division I baseball, covering both batting and pitching component statistics.

---

## Batting Model

The batting model estimates preseason true-talent distributions for individual hitters using Bayesian inference with conjugate priors. Each player is represented as a set of posterior distributions over four component statistics:

| Stat | Model | Description |
|------|-------|-------------|
| K% | Beta-Binomial | Strikeout rate |
| BB% | Beta-Binomial | Walk rate |
| ISO | Normal-Normal | Isolated power |
| BABIP | Normal-Normal | Batting average on balls in play |

**How it works:**
- Rate stats (K%, BB%) use a **Beta-Binomial** model. The prior is parameterized by a mean and a strength (equivalent plate appearances), and updates conjugately as successes/failures are observed.
- Continuous stats (ISO, BABIP) use a **Normal-Normal** model with a known likelihood variance estimated from historical data. The prior precision is scaled by the equivalent PA strength.
- Population priors are stratified by player age class (≤20, 21, 22, 23+) and seeded from historical NCAA FanGraphs data. Freshmen receive a separate prior reflecting their higher uncertainty.
- Conference strength is factored in via a `mean_conf_strength` weight that shrinks player estimates toward the conference mean.

**Population averages used as priors (overall):**
- K%: ~18.9% | BB%: ~11.2% | ISO: ~0.173 | BABIP: ~0.333

---

## Pitching Model

The pitching model mirrors the batting model structure and estimates preseason true-talent distributions for individual pitchers over four component statistics:

| Stat | Model | Description |
|------|-------|-------------|
| K% | Beta-Binomial | Strikeout rate (per batter faced) |
| BB% | Beta-Binomial | Walk rate (per batter faced) |
| HR/FB% | Normal-Normal | Home run per fly ball rate |
| BABIP | Normal-Normal | BABIP allowed |

**How it works:**
- The same conjugate prior framework as batting is used, with age-stratified population priors.
- HR/FB% and BABIP use Normal-Normal models with empirically estimated likelihood variances.
- A FIP constant (~3.98) and average fly ball rate (~38%) are stored in the model for downstream ERA/FIP estimation.
- As with batting, a conference strength coefficient shrinks pitchers toward their conference mean.

**Population averages used as priors (overall):**
- K%: ~21.0% | BB%: ~9.6% | HR/FB%: ~10.4% | BABIP: ~0.317

---

## Repository Structure

```
src/
  bayesian_batting/   # Batting model: priors, updates, aggregation, validation
  bayesian_pitching/  # Pitching model: priors, updates, aggregation, validation

models/
  fg_bayesian_batting/    # Fitted batting priors and validation results
  fg_bayesian_pitching/   # Fitted pitching priors and validation results
  pa_share_ridge_model.joblib  # Ridge regression model for PA share projection

data/                   # Input data (NCAA stats)
```
