This is a quant project that uses a statistical arbitrage strategy.

The purpose of this project is to gain familiarity with a quant strategy called statistical arbitrage, not necessarily to maximize P&L. This specific implementation of the strategy is adopted from a well-known research paper called "Statistical arbitrage in the US equities market" by Marco Avellaneda and Jeong-Hyun Lee.

This strategy uses Principal Component Analysis (PCA) and Ornstein-Uhlenbeck (OU) processes:
  - PCA uncovers latent economic factors that explain cross-sectional stock returns. These factors are used to calculate residuals and optimize portfolios via beta hedging.
  - OU process takes in the residuals to compute statistical S-scores as the likelihood of residual mean-reversion. The S-scores, along with parameters such as kappa (speed of mean-reversion), are used for signal generation.

The entire strategy including data visualizations, perfomance calculations, and Monte Carlo risk assessments can be executed in "main.py".
