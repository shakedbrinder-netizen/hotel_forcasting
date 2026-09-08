# Dan Hotels Occupancy Forecasting 

### Sumamry

This project delivers a production-ready AI Decision Support System (DSS) designed to forecast a 7-day rolling occupancy rate for Dan Tel Aviv and Dan Jerusalem. The core objective is to optimize HR and workforce scheduling based purely on historical patterns and localized calendars, establishing a highly accurate baseline without relying on current On-The-Books (OTB) reservations.

Following extensive evaluation, the system architecture was finalized using a Direct Multi-Step Forecasting approach. Shadow-mode backtesting across 2025 demonstrated exceptional resilience, achieving a True Operational MAE of ~2.3% for Dan Tel Aviv and ~16.1% for Dan Jerusalem.


### Data Engineering
To ensure the algorithms learn true market dynamics rather than anomalies, the raw PMS data is strictly sanitized:

* Structural Break Isolation: Macro-crises (e.g., Covid-19 lockdowns, October 7th escalations) are dynamically flagged. The training pipeline strictly filters out these periods to feed the algorithms exclusively with high-quality, representative behavioral data.
* Calendar Intelligence: Integration of the pyluach mathematical engine to encode Israeli holidays, Erev Chag, and Chol Hamoed as exogenous triggers.
* Cyclical Encoding: Linear dates are transformed into continuous mathematical waves (sine/cosine) to capture annual seasonality without abrupt year-end resets.


### Algorithmic Architecture: Direct Multi-Step Forecasting
Standard forecasting models suffer from "compounding error" when predicting 7 days ahead recursively (using tomorrow's guess to predict the day after). To circumvent this and capture the extreme weekend volatility of the Israeli market, we implemented a Direct Multi-Step Architecture.Instead of one model looping 7 times, the system trains 14 independent algorithms (7 for Dan Tel Aviv, 7 for Dan Jerusalem). Model $k$ is exclusively trained to map today's memory directly to day $T+k$, entirely bypassing the recursive snowball effect.

A. Dan Tel Aviv: ExtraTrees Regressor Ensemble
For the highly structured, corporate-driven market of Tel Aviv, we utilize Extremely Randomized Trees (ExtraTrees).
This ensemble learning method builds hundreds of independent decision trees. Unlike Random Forest, ExtraTrees selects split thresholds completely at random, significantly reducing variance and preventing overfitting on historical noise.

Mathematical Foundation:
The final prediction $\hat{y}$ is the aggregated average of all individual trees $f_m(x)$:      $$\hat{y} = \frac{1}{M} \sum_{m=1}^{M} f_m(x)$$

B. Dan Jerusalem: Nu-Support Vector Regression (NuSVR)
For the volatile, event-driven market of Jerusalem, we deployed NuSVR.
SVR projects data into a high-dimensional geometric space to find a function that fits the data within a specified tolerance margin ($\epsilon$). The parameter $\nu$ controls the upper bound on the fraction of margin errors and the lower bound of support vectors, making it highly resilient to sudden market shifts.

Mathematical Foundation:
The algorithm minimizes the structural risk by optimizing the hyperplane weights $w$: $$\min_{w,b,\xi,\xi^*} \frac{1}{2} \vert{}\vert{}w\vert{}\vert{}^2 - C \sum_{i=1}^{n} (\xi_i + \xi_i^*)$$
(Where $C$ is the regularization parameter, and $\xi_i, \xi_i^$ represent the deviation of samples falling outside the acceptable $\epsilon$-tube).*

4. Risk Management: 95% Confidence Intervals
To provide actionable risk-management metrics for HR Operations, the system outputs a 95% Statistical Confidence Interval alongside the main prediction, defining the probabilistic upper and lower bounds.
* For ExtraTrees (Dan Tel Aviv): The interval is calculated dynamically at inference time by measuring the standard deviation ($\sigma$) across the predictions of all the underlying estimators in the forest: $$CI_{95\%} = \hat{y} \pm 1.96 \times \sigma_{trees}$$

* For NuSVR (Dan Jerusalem): Since SVR is deterministic and lacks ensemble variance, the confidence bound is derived from the strictly evaluated out-of-sample historical Mean Absolute Error (MAE): $$CI_{95\%} = \hat{y} \pm \text{MAE}_{historical}$$


#### MLOps Deployment Structure
The system is entirely decoupled into a modern MLOps architecture:

1. Orchestrator (main.py): Filters data, executes the Multi-Step training, evaluates true MAE, and packages the 14 algorithms, scalers, and features into unified MLOps artifacts (.pkl).
2. Inference Engine (inference.py): The lightweight backend engine that loads the artifacts and computes the dynamic confidence bounds.
3. Frontend Dashboard (app.py): A Streamlit application allowing management to input the trailing 14-day occupancy and instantly visualize the 7-day statistical forecast without requiring technical expertise.

#### Phase 2 Roadmap: OTB Integration
Phase 1 has successfully pushed purely historical forecasting to its absolute mathematical limits, mastering baseline seasonality. To absorb unforeseen, non-cyclical events (such as last-minute corporate groups or sudden cancellations) and push Jerusalem's accuracy beyond 85%, the system requires forward-looking variables.

Phase 2 will focus on ingesting a sanitized On-The-Books (OTB) data stream from the central PMS. The current AI architecture is fully prepared to integrate these features as exogenous regressors.

    
