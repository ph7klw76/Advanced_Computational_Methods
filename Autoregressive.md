# Autoregressive (AR) models — a deeply intuitive, step-by-step guide

**One-sentence idea**: an AR model says “the next value is a baseline + weighted memory of the last few values + new surprise.”

Mathematically, for order $p$:

$$
y_t = c + \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \varepsilon_t,
$$

where $\varepsilon_t$ is white noise (zero mean, constant variance, no serial correlation).

---

## Step 0 — What problem does AR actually solve?

**Goal**: predict a time series that shows persistence (today looks like yesterday, but not exactly).

**Examples**: daily energy demand, semiconductor process outputs, sensor drift, macro indicators, many physical processes with inertia.

**When AR is a bad fit alone**: strong seasonality without seasonal terms; deterministic trends without differencing; big external drivers you’ve ignored.

---

## Step 1 — Build an intuition using AR(1)

Start with the simplest memory: AR(1)

$$
y_t = c + \phi y_{t-1} + \varepsilon_t.
$$

### Intuitive roles

- $c$: the baseline push (drift).
- $\phi$: the memory strength (how much of yesterday persists).
- $\varepsilon_t$: new information we couldn’t have known yesterday.

---

### Gravity picture (no noise, $\varepsilon_t = 0$)

Pick $c = 2$, $\phi = 0.6$.

- If $y_{t-1} = 10$:
  
$$
y_t = 2 + 0.6 \cdot 10 = 8 \quad \text{(pulled down toward 5)}.
$$

- If $y_{t-1} = 0$:
   
$$
y_t = 2 + 0 = 2 \quad \text{(pushed up toward 5)}.
$$

---

### Claim you can verify

If $|\phi| < 1$, the long-run mean is:

$$
\mu = \frac{c}{1 - \phi}.
$$

For $c = 2$, $\phi = 0.6 \Rightarrow \mu = 5$.

**Rationale**: take expectations on both sides and use $E[\varepsilon_t] = 0$.

---

### What $\phi$ “feels” like

- $|\phi| < 1$: shocks fade geometrically → stationary fluctuations around $\mu$.
- $\phi \approx 0$: almost no memory.
- $\phi \to 1$: unit root/random walk; shocks never fade (non-stationary).

---

### Another verifiable claim (variance if $|\phi| < 1$):

$$
\mathrm{Var}(y_t) = \frac{\sigma_\varepsilon^2}{1 - \phi^2}.
$$

**Rationale**: solve the variance recursion  
$\mathrm{Var}(y_t) = \phi^2 \mathrm{Var}(y_{t-1}) + \sigma_\varepsilon^2$

---

## Step 2 — Extend the memory: AR(p)

AR($p$) just says “today depends on the last $p$ yesterdays.”

$$
y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \varepsilon_t.
$$

Use it when one lag isn’t enough (e.g., physical relaxation taking several steps).

---

### Fingerprints in correlations

**ACF** (autocorrelation) for AR($p$): decays gradually (often geometrically or as a damped sine if complex roots).  
**PACF** (partial autocorrelation) for AR($p$): cuts off after lag $p$ (lags > $p$ near zero).  
→ This is a practical way to guess $p$ before formal selection.

---

## Step 3 — Make the series “AR-ready” (stationarity & preprocessing)

**Why we care**: the AR formulas above assume the series doesn’t drift in mean/variance.

Plot the series. Does it wander (trend) or change variance?

**Unit-root test** (Augmented Dickey–Fuller is common). If non-stationary:

- **Difference once**:
  
$$
x_t = y_t - y_{t-1}
$$

  → fit AR to $x_t$ (this is ARIMA).

- **Or model a deterministic trend**: include a time trend term.

**Seasonality?** Add seasonal AR lags (e.g., lag 7 for daily weekly pattern) → SARIMA or include seasonal dummies.

**Reasoning**: stationarity makes AR parameters stable and residuals well-behaved; without it, forecasts drift incorrectly.

---

## Step 4 — Choose $p$ in a principled way

A practical 3-tool routine:

- Look at PACF: first big cutoff suggests candidate $p$.
- Try a small grid $p \in \{1, \dots, 8\}$ (or more if data are long).
- Compare AIC/BIC: lower is better; BIC penalizes complexity more (helps avoid overfit).

**Reality check**: the chosen model should pass residual tests (next step). If not, revisit $p$, preprocessing, or add seasonal/exogenous terms.

---

## Step 5 — Fit the model (how parameters are learned)

Two common, verifiable routes:

1. **OLS regression**: regress $y_t$ on $1, y_{t-1}, \dots, y_{t-p}$ (drop first $p$ rows). This directly estimates $c, \phi_i$.
2. **Yule–Walker/Burg**: solve using sample autocorrelations; often similar results, popular in signal processing.

Either way you get:

- $\hat{c}, \hat{\phi}_i$
- residuals $\hat{\varepsilon}_t = y_t - \hat{c} - \sum \hat{\phi}_i y_{t-i}$
- estimate of noise variance $\hat{\sigma}_\varepsilon^2$

---

## Step 6 — Diagnose (trust but verify)

**Goal**: confirm the model captured all serial structure, leaving white-noise residuals.

- **Residual ACF**: should be near zero at all lags.
- **Ljung–Box test** on residuals at several lags (e.g., 10, 20): p-values not small → consistent with no remaining autocorrelation.
- **Stability check**: AR roots outside the unit circle (software reports). This implies the fitted process is stationary.
- **Homoskedasticity & distribution**: residual variance roughly constant; near-normal helps for interval accuracy but isn’t mandatory for point forecasts.

**If something fails**: increase/decrease $p$, difference the series, add seasonal lags, or include exogenous variables.

---

## Step 7 — Forecasting (the mechanism in action)

**One-step ahead (given data through time $T$):**

$$
\hat{y}_{T+1|T} = \hat{c} + \sum_{i=1}^{p} \hat{\phi}_i y_{T+1-i}
$$

**Multi-step ahead**: plug forecasts back in recursively:

$$
\hat{y}_{T+h|T} = \hat{c} + \sum_{i=1}^{p} \hat{\phi}_i \hat{y}_{T+h-i|T}
$$

*(replace unknown future lags by their forecasts)*

**Uncertainty**: forecast intervals widen with horizon because future steps depend on previous predicted values + accumulated noise variance. Packages compute these from $\hat{\sigma}_\varepsilon^2$ and the AR structure; you can verify by inspecting the reported standard errors per horizon.

---

## Step 8 — A tiny worked example you can reproduce

Simulate AR(1):  

$$
y_t = 2 + 0.6 y_{t-1} + \varepsilon_t,\quad \varepsilon_t \sim \mathcal{N}(0, 1)
$$

- **Theoretical mean**:
  
$$
\frac{2}{1 - 0.6} = 5
$$

- **Theoretical variance**:
  
$$
\frac{1}{1 - 0.6^2} = \frac{1}{0.64} \approx 1.5625
$$

**What you’ll typically observe after fitting**:

- Estimated $\hat{\phi} \approx 0.6$, $\hat{c} \approx 0.8$  
  (many packages use intercept $c$ directly; implied mean is $\hat{c}/(1 - \hat{\phi})$, which should be near 5).
- Residual ACF shows no structure; Ljung–Box p-values are not significant.
- PACF of the original series cuts off after lag 1, consistent with AR(1).

---

## Step 9 — Interpreting coefficients (and not over-interpreting)

- Magnitude of $\phi_i$: tells you how much each past step matters.
- Sign pattern: positive/negative lags can indicate overshoot/oscillation (complex AR roots → damped oscillations in ACF).
- Sum of $\phi_i$: affects the long-run mean
  
$$
\mu = \frac{c}{1 - \sum_i \phi_i} \quad \text{(if stationary)}
$$

**Caution**: coefficients are conditional on preprocessing. If you difference the series, coefficients describe dynamics of changes, not levels.

---

## Step 10 — Common pitfalls and surgical fixes

| Symptom | Likely cause | Fix |
|--------|---------------|-----|
| Residual ACF still shows spikes | $p$ too small or wrong structure | Increase $p$, check PACF again, consider ARMA/SARIMA |
| Clear weekly/yearly cycles remain | Seasonality not modeled | Add seasonal AR lags (SARIMA) or seasonal dummies |
| Trend/drift not captured | Unit root / non-stationarity | Difference once (ARIMA), or include trend |
| Forecasts explode or wobble | Unstable (unit circle roots inside) | Refit; ensure stationarity; reduce $p$ |
| Big shocks at known dates | Structural break | Split sample, add regime dummies, or regime-switching models |
| External factors obviously matter | Missing regressors | Use ARX/ARIMAX with those drivers |

---

## Step 11 — AR vs MA vs ARMA vs ARIMA (quick clarity)

- **AR**: depends on past values of $y$.
- **MA**: depends on past shocks $\varepsilon$ (captures short-memory noise patterns).
- **ARMA**: both (often more flexible).
- **ARIMA**: ARMA on a differenced series to handle non-stationary levels.
- **SARIMA**: ARIMA + seasonal AR/MA terms.

If AR residuals show a distinct ACF “cutoff” (not tailing), a MA component may be missing.

---

## Step 12 — A crisp, reproducible workflow (use this as a checklist)

1. Visualize the series. Detect trend/seasonality/outliers.
2. **Stationarity check**: ADF test; difference or add trend/seasonal terms if needed.
3. **ACF/PACF**: hypothesize $p$.
4. Fit candidates over a small grid; compare AIC/BIC.
5. Diagnostics: residual ACF, Ljung–Box, stability (roots).
6. Refine (adjust $p$/seasonal/exogenous terms) until residuals look white.
7. Forecast and report intervals; sanity-check against domain knowledge.
8. Out-of-sample validation (rolling origin if possible) to confirm generalization.

---

## Mini “why” proofs you can check

**Mean formula (AR(1))**:  
Take expectation:  

$E[y_t] = c + \phi E[y_{t-1}] \Rightarrow \mu = c + \phi \mu \Rightarrow \mu = \frac{c}{1 - \phi}$  
(if $|\phi| < 1$)

**Variance formula (AR(1))**: 

$\mathrm{Var}(y_t) = \phi^2 \mathrm{Var}(y_{t-1}) + \sigma_\varepsilon^2 \Rightarrow v = \phi^2 v + \sigma_\varepsilon^2 \Rightarrow v = \frac{\sigma_\varepsilon^2}{1 - \phi^2}$

**PACF cutoff (AR(p))**:  
By construction, once you control for the first $p$ lags, higher lags add no incremental linear information; hence partial correlations at lags $>p$ are (asymptotically) zero.  
You can verify empirically after fitting.

---

## Practical tips from experience

- Prefer **BIC** if you want a simpler, robust model; prefer **AIC** if you’ll ensemble or you have lots of data.
- **Scale matters**: if the series has huge magnitude, standardize before fitting for numerical stability (interpretation is unchanged in original units through inverse transform).
- **Cross-validation for time series**: use rolling (expanding or sliding) windows, not random K-fold.
- **Report both**: parameters and residual diagnostics; parameters alone can mislead.

---

## Cheat sheet (printable)

**Model**:  

$$
y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \varepsilon_t
$$

- **Stationarity**: AR roots outside unit circle; AR(1) needs $|\phi| < 1$.
- **Mean (stationary)**:
  
$$
\mu = \frac{c}{1 - \sum_i \phi_i} \quad \text{(AR(1): } \mu = \frac{c}{1 - \phi} \text{)}
$$

- **ACF/PACF**: AR($p$) → PACF cuts at $p$, ACF tails.
- **Fit**: OLS or Yule–Walker; choose $p$ via AIC/BIC + PACF.
- **Check**: residual ACF ≈ 0; Ljung–Box p-values not small; stability holds.
- **Forecast**: recurse; intervals widen with horizon.

Here are simple, real-life physics examples where an AR model (often with tiny $p$) works well. For each: what to model → preprocess → typical $p$ → why it’s useful. All claims are verifiable by plotting ACF/PACF, checking Ljung–Box on residuals, and confirming AR roots are outside the unit circle.

---

### 1) Lab room temperature (short-term drift)

- **Model**: minute-by-minute temperature $T_t$.
- **Preprocess**: remove daily cycle (e.g., subtract 24-h moving average or seasonal dummy).
- **Typical $p$**: 1–2 (AR(1) often sufficient).
- **Use**: predict near-future temp for sensitive experiments; detect HVAC anomalies when residuals cease to be white.

---

### 2) Optical table/cryostat vibration (low-freq drift)

- **Model**: RMS acceleration/velocity measured each second.
- **Preprocess**: low-pass to isolate drift; remove mean.
- **Typical $p$**: 1–3.
- **Use**: forecast sub-minute vibration levels to pause a measurement just before spikes.

---

### 3) Thin-film deposition rate (QCM) stabilization

- **Model**: deposition rate Hz/s (or thickness increment per second).
- **Preprocess**: difference once if there’s a slow ramp; exclude actuator steps.
- **Typical $p$**: 1–2.
- **Use**: anticipate rate overshoot after source power changes; feedforward damping (ARX if you include power as an input).

---

### 4) Vacuum chamber pressure settling

- **Model**: log-pressure sampled every few seconds during pump-down near steady state.
- **Preprocess**: focus on the residual after fitting an exponential decay (the residual is often stationary).
- **Typical $p$**: 1–2.
- **Use**: predict time to reach target pressure; flag leaks when residual structure persists.

---

### 5) Laser power noise (photodiode)

- **Model**: power (or normalized intensity) at kHz → downsample/average to ~100 Hz.
- **Preprocess**: remove slow drift (high-pass or polynomial detrend).
- **Typical $p$**: 1–4 (sometimes ARMA fits better; start AR).
- **Use**: short-horizon prediction for adaptive exposure control; verify whitened residuals → healthy laser.

---

### 6) Interferometer stage drift (nm-level)

- **Model**: stage position error every second.
- **Preprocess**: subtract reference (temperature sensor) or fit & remove linear trend.
- **Typical $p$**: 1–3.
- **Use**: forecast next-minute drift to schedule recalibration; detect step changes (breaks) when residual variance jumps.

---

### 7) Semiconductor process metrology (e.g., line width)

- **Model**: wafer-to-wafer deviation from target (residual).
- **Preprocess**: remove lot trends; center by target spec.
- **Typical $p$**: 1–2.
- **Use**: predict next-wafer bias and apply run-to-run correction (EWMA ≈ AR(1) intuition).

---

### 8) Hall probe / magnetometer baseline wander

- **Model**: baseline field reading at 1–10 Hz in a “quiet” shielded setup.
- **Preprocess**: remove obvious steps (switching events); detrend.
- **Typical $p$**: 1–2.
- **Use**: anticipate baseline drift to improve subtraction for tiny signal detection.

---

### 9) Thermal camera pixel time series (one pixel)

- **Model**: pixel temperature (or radiance) per frame under constant scene.
- **Preprocess**: remove global scene average to isolate pixel drift.
- **Typical $p$**: 1–2.
- **Use**: forecast and correct fixed-pattern/temporal noise; improve detection of small hotspots.

---

### 10) Wind speed (short-term, 1–5 min horizon)

- **Model**: 1-min mean wind speed.
- **Preprocess**: remove diurnal cycle; cap outliers (gusts).
- **Typical $p$**: 1–3.
- **Use**: anticipate near-term loads on towers/telescopes; schedule exposures when blur risk is low.

---

### 11) River flow anomalies (hydrology)

- **Model**: daily flow anomaly = flow − seasonal average.
- **Preprocess**: subtract climatology; optionally variance-stabilize (Box–Cox).
- **Typical $p$**: 1–2.
- **Use**: short-term anomaly forecasting for operational decisions; residual whiteness indicates good seasonal removal.

---

### 12) Plasma etch endpoint OES intensity (steady phase)

- **Model**: intensity residual around plateau.
- **Preprocess**: window the steady segment; subtract local mean.
- **Typical $p$**: 1–2.
- **Use**: predict imminent endpoint deviation; early alarms when residual ACF spikes reappear.

---

### 13) Calorimetry baseline (DSC) in an isothermal hold

- **Model**: heat-flow baseline after initial transients.
- **Preprocess**: discard first minutes; detrend any slow curvature.
- **Typical $p$**: 1.
- **Use**: detect subtle exotherms/endotherms as departures from an AR(1) baseline.

---

### 14) Battery terminal voltage under constant load (short-term)

- **Model**: voltage drift at fixed current over minutes.
- **Preprocess**: difference if there’s a clear monotonic trend; otherwise center.
- **Typical $p$**: 1–2.
- **Use**: anticipate near-term sag for control; residual autocorrelation hints at unmodeled thermal coupling.

---

### 15) Spectrometer dark current (per integration)

- **Model**: dark signal vs. time for a given integration time.
- **Preprocess**: temperature-compensate first (if sensor temp available), then center.
- **Typical $p$**: 1.
- **Use**: predict & subtract dark fluctuations; whitened residuals validate correction.

```python
# ============================================================
# AR Modeling for (1) Room Temperature & (2) Vibration Drift
# Backward-compatible with older statsmodels/matplotlib
# Requirements: numpy, pandas, matplotlib, statsmodels
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL


# ----------------------------
# Utilities
# ----------------------------
def choose_ar_order(y, max_lag=8):
    """
    Fit AutoReg models for lags 1..max_lag and choose the one with lowest BIC.
    Returns (best_lag, fitted_model).
    """
    best_model = None
    best_lag = None
    best_bic = np.inf

    y = pd.Series(y).dropna().values  # ensure no NaNs

    for p in range(1, max_lag + 1):
        try:
            model = AutoReg(y, lags=p, old_names=False).fit()
            if model.bic < best_bic:
                best_bic = model.bic
                best_model = model
                best_lag = p
        except Exception:
            continue

    return best_lag, best_model


def _compute_ar_roots(phi):
    """
    Compute roots of the AR characteristic polynomial:
       1 - phi1*z - phi2*z^2 - ... - phip*z^p = 0
    Stationarity requires all |roots| > 1.
    """
    poly = np.r_[1.0, -np.asarray(phi)]  # [1, -phi1, -phi2, ..., -phip]
    return np.roots(poly)


def check_whiteness_and_stability(model, lb_lags=(10, 20)):
    """
    Print Ljung-Box results and check AR roots for stability.
    Compatible with older statsmodels.
    """
    resid = model.resid
    lb = acorr_ljungbox(resid, lags=list(lb_lags), return_df=True)
    print("\nLjung–Box test on residuals:")
    print(lb)

    # Try native arroots, else compute from params
    phi = _extract_phi_from_model(model)
    try:
        roots = model.arroots  # may not exist in older versions
        magnitudes = np.abs(roots)
    except Exception:
        if phi is not None and len(phi) > 0:
            roots = _compute_ar_roots(phi)
            magnitudes = np.abs(roots)
        else:
            roots = []
            magnitudes = []

    if len(magnitudes) > 0:
        print("\nAR roots (magnitudes):", np.round(magnitudes, 4))
        print("Stable (all roots > 1)?", bool(np.all(magnitudes > 1.0)))
    else:
        print("\nCould not compute AR roots for stability check.")


def plot_series(title, x, y, xlabel="Time", ylabel="Value"):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()


def _safe_pacf(y, nlags=40):
    """
    Robust PACF computation across statsmodels versions.
    Prefers 'ywmle', falls back to 'ywm'.
    """
    try:
        return pacf(y, nlags=nlags, method="ywmle")
    except Exception:
        return pacf(y, nlags=nlags, method="ywm")


def plot_acf_pacf(y, nlags=40, title_prefix=""):
    # ACF
    acf_vals = acf(y, nlags=nlags, fft=True)
    plt.figure()
    plt.stem(range(len(acf_vals)), acf_vals)  # no use_line_collection
    plt.title(f"{title_prefix}ACF")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.tight_layout()

    # PACF
    pacf_vals = _safe_pacf(y, nlags=nlags)
    plt.figure()
    plt.stem(range(len(pacf_vals)), pacf_vals)
    plt.title(f"{title_prefix}PACF")
    plt.xlabel("Lag")
    plt.ylabel("Partial Autocorrelation")
    plt.tight_layout()


def plot_residual_acf(resid, nlags=40, title="Residual ACF"):
    r_acf = acf(resid, nlags=nlags, fft=True)
    plt.figure()
    plt.stem(range(len(r_acf)), r_acf)
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.tight_layout()


def _extract_c_phi_from_params(params):
    """
    From AutoReg .params -> intercept c and AR coefficients phi[1..p].
    statsmodels packs [intercept, phi1, phi2, ..., phip].
    """
    params = np.asarray(params)
    c = params[0]
    phi = params[1:]
    return c, phi


def _extract_phi_from_model(model):
    try:
        _, phi = _extract_c_phi_from_params(model.params)
        return phi
    except Exception:
        return None


def _psi_weights(phi, steps):
    """
    Compute psi-weights for AR(p) up to psi_{steps-1} using recursion:
      psi_0 = 1
      psi_h = sum_{j=1}^{min(h,p)} phi_j * psi_{h-j}
    """
    p = len(phi)
    psi = np.zeros(steps)
    psi[0] = 1.0
    for h in range(1, steps):
        upto = min(h, p)
        s = 0.0
        for j in range(1, upto + 1):
            s += phi[j - 1] * psi[h - j]
        psi[h] = s
    return psi


def _z_from_alpha(alpha):
    """
    Get z-quantile for two-sided (1-alpha) CI.
    Try scipy if available; else common defaults for 95%/90%.
    """
    try:
        from scipy.stats import norm
        return float(norm.ppf(1.0 - alpha / 2.0))
    except Exception:
        if abs(alpha - 0.05) < 1e-9:
            return 1.96
        if abs(alpha - 0.10) < 1e-9:
            return 1.6448536269514722  # ~1.645
        # fallback approximate
        return 1.96


def forecast_with_ci_compat(y, model, steps=60, alpha=0.05):
    """
    Backward-compatible multi-step forecast for AR(p) with CIs.
    - Uses recursive mean forecasts with fitted (c, phi).
    - CI via innovation variance * sum(psi^2) up to horizon (Gaussian approx).
    Inputs:
      y: 1D array used to fit the model (last p values needed)
      model: AutoRegResults
    Returns:
      mean (length steps), lower, upper (arrays)
    """
    y = np.asarray(y)
    c, phi = _extract_c_phi_from_params(model.params)
    p = len(phi)
    if p == 0:
        raise ValueError("Model has no AR lags; cannot forecast AR(p) with p=0.")

    # Innovation variance estimate (use residual variance)
    # model.sigma2 exists in some versions; if not, compute from residuals
    try:
        sigma2 = float(model.sigma2)
    except Exception:
        resid = model.resid
        # ddof ~ number of estimated params; conservative choice:
        sigma2 = float(np.var(resid, ddof=p + 1))

    # Prepare recursive container seeded with last p observed y
    history = list(y[-p:])
    fc_mean = np.zeros(steps)

    for h in range(steps):
        # yhat = c + sum phi_i * history[-i]
        yhat = c
        for i in range(1, p + 1):
            yhat += phi[i - 1] * history[-i]
        fc_mean[h] = yhat
        history.append(yhat)  # use forecast as future lag

    # Standard errors via psi-weights
    psi = _psi_weights(phi, steps=steps)
    # cumulative sum of psi^2 for each horizon h: sum_{i=0}^{h-1} psi_i^2
    psi2_cumsum = np.cumsum(psi**2)
    se = np.sqrt(sigma2 * psi2_cumsum)

    z = _z_from_alpha(alpha)
    lower = fc_mean - z * se
    upper = fc_mean + z * se
    return fc_mean, lower, upper


# ============================================================
# (1) LAB ROOM TEMPERATURE (short-term drift)
# Goal: Remove daily cycle -> AR(p) on anomaly -> diagnose -> forecast
# ============================================================
def run_room_temperature_pipeline(
    df=None,
    period_minutes=1440,
    forecast_minutes=60,
    use_stl=True
):
    """
    df: DataFrame with columns ['timestamp','value'] at ~1-min resolution.
        If None, synthetic data are generated.
    period_minutes: diurnal period at 1-min sampling (1440 for daily).
    forecast_minutes: horizon for forecasts (1-min steps).
    use_stl: if True use STL for seasonal removal; else rolling mean.
    """
    if df is None:
        # --- Synthetic data with daily cycle + AR(1) anomaly ---
        rng = pd.date_range("2024-01-01", periods=1440 * 3, freq="min")  # 3 days, 1-min
        base_temp = 23.0
        daily_amp = 1.5
        daily = base_temp + daily_amp * np.sin(2 * np.pi * np.arange(len(rng)) / period_minutes)

        # AR(1) anomaly
        np.random.seed(42)
        phi = 0.6
        c = 0.0
        eps = np.random.normal(scale=0.15, size=len(rng))
        anomaly = np.zeros(len(rng))
        for t in range(1, len(rng)):
            anomaly[t] = c + phi * anomaly[t - 1] + eps[t]

        value = daily + anomaly
        df = pd.DataFrame({"timestamp": rng, "value": value})

    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    # Plot raw temperature
    plot_series("Room temperature (raw)", df["timestamp"], df["value"], ylabel="°C")

    # --- Remove daily cycle ---
    s = pd.Series(df["value"].values, index=df["timestamp"])
    if use_stl:
        stl = STL(s, period=period_minutes, robust=True).fit()
        seasonal = stl.seasonal
        anomaly = s - seasonal  # remove diurnal pattern
        anomaly = anomaly - anomaly.mean()  # center
    else:
        seasonal_est = s.rolling(
            window=period_minutes,
            min_periods=int(0.7 * period_minutes),
            center=True
        ).mean()
        anomaly = (s - seasonal_est).dropna()
        anomaly = anomaly - anomaly.mean()

    # Plot anomaly (stationary candidate)
    plot_series("Temperature anomaly (diurnal removed)", anomaly.index, anomaly.values, ylabel="°C (anomaly)")

    # ACF/PACF for order hint
    plot_acf_pacf(anomaly.values, nlags=60, title_prefix="Temp anomaly: ")

    # --- Fit AR(p) by BIC ---
    p, model = choose_ar_order(anomaly.values, max_lag=8)
    print(f"\n[Room Temp] Chosen AR order by BIC: p={p}")
    print(model.summary())

    # Diagnostics
    check_whiteness_and_stability(model, lb_lags=(10, 20))
    plot_residual_acf(model.resid, nlags=60, title="Room Temp residual ACF")

    # --- Forecast next 'forecast_minutes' (1-min steps), compat mode ---
    fc_mean, fc_lo, fc_hi = forecast_with_ci_compat(anomaly.values, model, steps=forecast_minutes, alpha=0.05)
    fc_index = pd.date_range(anomaly.index[-1] + pd.Timedelta(minutes=1), periods=forecast_minutes, freq="min")

    # Plot forecast (focus on recent context)
    plt.figure()
    plt.plot(anomaly.index[-200:], anomaly.values[-200:])
    plt.plot(fc_index, fc_mean)
    plt.fill_between(fc_index, fc_lo, fc_hi, alpha=0.2)
    plt.title("Room Temp anomaly: last 200 min + forecast")
    plt.xlabel("Time")
    plt.ylabel("°C (anomaly)")
    plt.tight_layout()

    return {"data": df, "anomaly": anomaly, "model": model, "forecast": (fc_index, fc_mean, fc_lo, fc_hi)}


# ============================================================
# (2) OPTICAL TABLE / CRYOSTAT VIBRATION (low-frequency drift)
# Goal: Smooth to isolate drift -> remove mean -> AR(p) -> diagnose -> forecast
# ============================================================
def run_vibration_pipeline(
    df=None,
    smooth_seconds=30,
    forecast_seconds=30
):
    """
    df: DataFrame with columns ['timestamp','value'] at ~1 Hz (RMS accel/vel).
        If None, synthetic data are generated (slow drift + small AR component).
    smooth_seconds: moving-average window to isolate drift (in samples if 1 Hz).
    forecast_seconds: horizon for forecasts (1-s steps).
    """
    if df is None:
        # --- Synthetic drift + AR(2) anomaly around drift ---
        n = 2000  # seconds
        rng = pd.date_range("2024-01-01", periods=n, freq="S")
        np.random.seed(123)

        # Slow drift (e.g., exponential relaxation + tiny sinusoid)
        t = np.arange(n)
        drift = 0.5 * np.exp(-t / 1500) + 0.02 * np.sin(2 * np.pi * t / 300)

        # AR(2) small anomaly
        phi1, phi2 = 0.5, -0.2
        eps = np.random.normal(scale=0.02, size=n)
        anomaly = np.zeros(n)
        for i in range(2, n):
            anomaly[i] = phi1 * anomaly[i - 1] + phi2 * anomaly[i - 2] + eps[i]

        value = drift + anomaly + 0.005 * np.random.normal(size=n)
        df = pd.DataFrame({"timestamp": rng, "value": value})

    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    # Plot raw vibration
    plot_series("Vibration (RMS, raw)", df["timestamp"], df["value"], ylabel="RMS (a.u.)")

    # --- Smooth to isolate low-frequency component (drift) ---
    s = pd.Series(df["value"].values, index=df["timestamp"])
    smooth = s.rolling(
        window=smooth_seconds,
        min_periods=int(0.8 * smooth_seconds),
        center=True
    ).mean()

    # Anomaly around smoothed drift
    anomaly = (s - smooth).dropna()
    anomaly = anomaly - anomaly.mean()

    # Plot anomaly
    plot_series("Vibration anomaly (after smoothing)", anomaly.index, anomaly.values, ylabel="RMS anomaly (a.u.)")

    # ACF/PACF for order hint
    plot_acf_pacf(anomaly.values, nlags=60, title_prefix="Vibration anomaly: ")

    # --- Fit AR(p) by BIC ---
    p, model = choose_ar_order(anomaly.values, max_lag=8)
    print(f"\n[Vibration] Chosen AR order by BIC: p={p}")
    print(model.summary())

    # Diagnostics
    check_whiteness_and_stability(model, lb_lags=(10, 20))
    plot_residual_acf(model.resid, nlags=60, title="Vibration residual ACF")

    # --- Forecast next 'forecast_seconds' (1-s steps), compat mode ---
    fc_mean, fc_lo, fc_hi = forecast_with_ci_compat(anomaly.values, model, steps=forecast_seconds, alpha=0.05)
    fc_index = pd.date_range(anomaly.index[-1] + pd.Timedelta(seconds=1), periods=forecast_seconds, freq="S")

    # Plot forecast (last 300 s for context)
    plt.figure()
    plt.plot(anomaly.index[-300:], anomaly.values[-300:])
    plt.plot(fc_index, fc_mean)
    plt.fill_between(fc_index, fc_lo, fc_hi, alpha=0.2)
    plt.title("Vibration anomaly: last 300 s + forecast")
    plt.xlabel("Time")
    plt.ylabel("RMS anomaly (a.u.)")
    plt.tight_layout()

    return {"data": df, "anomaly": anomaly, "model": model, "forecast": (fc_index, fc_mean, fc_lo, fc_hi)}


if __name__ == "__main__":
    # --- Option A: Use your CSV data ---
    # For room temperature (1-min data):
    # temp_df = pd.read_csv("room_temperature.csv", parse_dates=["timestamp"])
    # run_room_temperature_pipeline(temp_df, period_minutes=1440, forecast_minutes=60, use_stl=True)

    # For vibration (1 Hz data):
    # vib_df = pd.read_csv("vibration_rms.csv", parse_dates=["timestamp"])
    # run_vibration_pipeline(vib_df, smooth_seconds=30, forecast_seconds=30)

    # --- Option B: Run synthetic demos (no files needed) ---
    out_temp = run_room_temperature_pipeline(
        df=None,
        period_minutes=1440,
        forecast_minutes=60,
        use_stl=True
    )
    out_vib = run_vibration_pipeline(
        df=None,
        smooth_seconds=30,
        forecast_seconds=30
    )

    plt.show()

```
