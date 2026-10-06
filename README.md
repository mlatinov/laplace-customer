# laplace-customer

Customer-analytics models for [Laplace](https://github.com/mlatinov/laplace): churn for subscription businesses, purchase-and-spend models for non-contractual businesses, and customer lifetime value. Everything is written so that per-customer quantities (churn risk, probability of being active, expected purchases, value) come out as full posterior distributions, not point estimates.

The package targets two kinds of business:

- **Contractual** (subscriptions, SaaS, memberships): you see the cancellation. Lifetime is a censored duration, so the `churn_*` functions build on [`laplace-survival`](https://github.com/mlatinov/laplace-survival).
- **Non-contractual** (retail, e-commerce): customers never announce leaving; they just stop buying. The `bgnbd_*`, `pnbd_*` and `gg_*` functions infer who is still active from the pattern of purchases.

## Install

```bash
laplace add customer --git https://github.com/mlatinov/laplace-customer --tag 0.1.0 --subdir laplace
```

```stan
library {
  import customer
}
```

Dependencies (`survival 0.1.1`, `splines 0.1.0`, `cluster 0.1.0`) are resolved automatically. Laplace imports are private: if your own model calls `survival::` or `cluster::` directly, add and import them yourself as well. Laplace unifies the versions.

Some expectations use Stan's built-in `hypergeometric_2F1`, so use a recent CmdStan.

## What's inside

### Utilities

| Function | Returns |
|---|---|
| `quad_nodes`, `quad_weights` | Composite Gauss–Legendre rule on `[a, b]` |
| `log_quad` | Log of an integral from log-integrand values (no underflow) |
| `disc_factor`, `cont_disc` | Discount factors, per-period or continuous rate |
| `h2f1` | Gauss hypergeometric series (fallback and test oracle for `hypergeometric_2F1`) |

### Churn: contractual customers

All families: Gompertz, Makeham, log-logistic (log-logistic is accelerated failure time, so covariates act on the time scale, not the hazard).

**Active customers today**

| Function | Returns |
|---|---|
| `churn_<family>_delta_H` | Increase in cumulative hazard over the next `s` time units, given current tenure |
| `churn_cond_surv`, `churn_prob` | Probability still subscribed / probability of cancelling in the next `s`, given tenure so far |
| `churn_<family>_rmrl` | Expected subscribed time within a horizon (restricted mean residual lifetime) |
| `churn_value_discrete`, `churn_gompertz_value` | Expected discounted margin from an active subscriber |

**Mixture cure: customers who never churn**

| Function | Returns |
|---|---|
| `churn_cure_loglik`, `churn_cure_ll` | Pointwise / summed likelihood from any hazard |
| `churn_cure_<family>_lpdf`, `_rng` | Family versions and simulators |
| `churn_cure_p_loyal` | Probability an active customer is a never-churner (a loyalty score) |

**Gamma frailty: unmeasured loyalty**

| Function | Returns |
|---|---|
| `churn_frailty_loglik`, `churn_frailty_ll` | Pointwise / summed likelihood from any hazard |
| `churn_frailty_<family>_lpdf`, `_rng` | Family versions and simulators |
| `churn_frailty_score` | Posterior hazard multiplier per customer (below 1 = more loyal than covariates suggest) |
| `churn_frailty_cond_surv` | Conditional survival that accounts for survivor selection |

**Competing risks: why customers leave**

| Function | Returns |
|---|---|
| `churn_competing_loglik`, `churn_competing_ll` | Cause-specific likelihood from any hazards |
| `churn_competing_cif` | Cumulative incidence per cause from any hazards |
| `churn_competing_<family>_lpdf`, `_cif`, `_rng` | Gompertz and Makeham versions |

**Latent segments: customer types with different churn curves**

| Function | Returns |
|---|---|
| `churn_segment_L`, `churn_segment_<family>_L` | N × K matrix of per-segment log likelihoods, to feed `cluster::mixture_lpdf` |
| `churn_segment_cond_surv` | Membership-weighted conditional survival per customer |

### CLV: non-contractual purchases and value

**BG/NBD** (dropout can happen right after a purchase)

| Function | Returns |
|---|---|
| `bgnbd_lpmf`, `bgnbd_loglik`, `bgnbd_weighted_lpmf` | Likelihood; weighted version for compressed unique `(x, t_x, T)` rows |
| `bgnbd_log_p_alive` | Log probability each customer is still active |
| `bgnbd_expected_uncond` | Expected repeat purchases of a new customer |
| `bgnbd_expected_cond` | Expected future purchases of each existing customer |
| `bgnbd_indiv_loglik` | Likelihood given each customer's own rate and dropout (hierarchical models with covariates) |
| `bgnbd_rng`, `bgnbd_indiv_rng` | Simulators for prior/posterior predictive checks and SBC |

**Pareto/NBD, individual level** (dropout can happen at any time)

| Function | Returns |
|---|---|
| `pnbd_indiv_lpmf`, `pnbd_indiv_loglik` | Likelihood given each customer's purchase and lapse rates |
| `pnbd_log_p_alive` | Log probability each customer is still active |
| `pnbd_expected_cond` | Expected future purchases |
| `pnbd_det` | Discounted expected transactions (infinite horizon) |
| `pnbd_rng`, `pnbd_indiv_rng` | Simulators |

**Spend and value**

| Function | Returns |
|---|---|
| `gg_lpdf`, `gg_loglik` | Gamma-gamma likelihood of average spend (repeat buyers only) |
| `gg_expected_spend` | Expected future transaction value, shrunk toward the population mean |
| `clv_discrete` | Discounted CLV from any matrix of cumulative expected purchases |

Every function carries `@brief`, `@param`, `@return` and `@math` doc comments with the exact formula it computes.

## Conventions

- **Function kinds.** `_lpdf` / `_lpmf` return a summed log density for `target +=` or `~`; `_loglik` returns the pointwise vector for LOO; `_ll` is the summed generic core; `_rng` simulates (only in `transformed data` / `generated quantities`).
- **Generic core + family wrapper.** Stan functions can't take functions as arguments, so each churn extension has a core that takes evaluated `log_h` and `H` (use any hazard you like) and thin Gompertz/Makeham/log-logistic wrappers.
- **One time unit.** Tenure, recency, age, horizons and discount rates must all use the same unit (weeks or months).
- **Shared vs per-customer.** A `real` parameter is shared, a `vector` one is per customer (covariates).

## Data you need

**Subscriptions** (one row per customer): tenure `t` (> 0), event indicator `d` (1 = cancelled, 0 = still active), optional `cause` (0 = active), covariates. Keep active customers: they are censored, not missing.

**Purchases** (one row per customer, from a transaction log): `x` = repeat purchases (first purchase excluded), `t_x` = time of last repeat purchase since the first, `T` = time from first purchase to the data cut-off, `mbar` = average repeat-purchase value (only for `x >= 1`).

## Examples

### Subscription churn with never-churners

```stan
library {
  import customer
  import survival
}
data {
  int<lower=1> N; vector<lower=0>[N] t; vector<lower=0, upper=1>[N] d;
  int<lower=1> P; matrix[N, P] X; int<lower=1> Q; matrix[N, Q] Z;
}
parameters {
  real<lower=0> gamma; vector[P] beta; real kappa0; vector[Q] kappa;
}
model {
  target += customer::churn_cure_gompertz_lpdf(t | gamma, X * beta, kappa0 + Z * kappa, d);
  // priors ...
}
generated quantities {
  vector[N] H = survival::srv_gompertz_cumulative_hazard(t, gamma, X * beta);
  vector[N] p_loyal = customer::churn_cure_p_loyal(kappa0 + Z * kappa, H);   // use where d == 0
}
```

### BG/NBD + gamma-gamma CLV

```stan
library { import customer }
data {
  int<lower=1> N; array[N] int<lower=0> x; vector<lower=0>[N] t_x; vector<lower=0>[N] T;
  vector<lower=0>[N] mbar;                       // any positive placeholder where x == 0
  int<lower=0> N_rep; array[N_rep] int<lower=1, upper=N> idx_rep;
  int<lower=1> K; real<lower=0> period; real<lower=0, upper=1> margin; real<lower=0> d;
}
parameters {
  real<lower=0> r; real<lower=0> alpha; real<lower=1> a; real<lower=0> b;
  real<lower=0> p; real<lower=1> q; real<lower=0> g;
}
model {
  x ~ customer::bgnbd(t_x, T, r, alpha, a, b);
  mbar[idx_rep] ~ customer::gg(x[idx_rep], p, q, g);
  // priors ...
}
generated quantities {
  vector[N] p_alive = exp(customer::bgnbd_log_p_alive(x, t_x, T, r, alpha, a, b));
  vector[N] spend = customer::gg_expected_spend(mbar, x, p, q, g);   // x == 0 -> population mean
  matrix[N, K] C;
  for (k in 1:K) C[, k] = customer::bgnbd_expected_cond(k * period, x, t_x, T, r, alpha, a, b);
  vector[N] clv = customer::clv_discrete(C, spend, margin, d);
  real equity = sum(clv);
}
```

## Good to know

- BG/NBD gives `P(alive) = 1` to every customer with `x = 0` (dropout can only follow a purchase). Don't rank one-time buyers by it.
- Expected-purchase functions need `a > 1` in BG/NBD; check the posterior before forecasting.
- Gamma-gamma assumes spend is independent of purchase frequency; check `corr(x, mbar)` first.
- In cure models with a Gompertz hazard, constrain `gamma > 0`: a negative slope already implies a never-churn fraction.
- Quadrature-based quantities (RMRL, continuous value, CIFs) carry a small bias; double the number of panels to check it.

## Not in 0.1.0

Cohort retention (sBG, beta-discrete-Weibull, DERL), BG/BB for discrete purchase occasions, M-spline baseline hazards, discrete-time churn, the latent-lifetime purchase model, model-based RFM segmentation, and convenience CLV wrappers are specified but planned for later releases.

## References

- Fader, Hardie & Lee (2005). "Counting your customers" the easy way. *Marketing Science* 24(2).
- Fader, Hardie & Lee (2005). RFM and CLV: using iso-value curves. *Journal of Marketing Research* 42(4).
- Schmittlein, Morrison & Colombo (1987). Counting your customers. *Management Science* 33(1).
- Abe (2009). "Counting your customers" one by one. *Marketing Science* 28(3).
- Fader & Hardie (2007). How to project customer retention. *Journal of Interactive Marketing* 21(1).
- Vaupel, Manton & Stallard (1979). Heterogeneity in individual frailty. *Demography* 16(3).