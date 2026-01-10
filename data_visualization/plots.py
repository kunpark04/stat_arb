import numpy as np
import matplotlib.pyplot as plt

def plot_equity(rets, cum_rets):
    # EQUITY CURVE
    # Total return
    total_ret = (cum_rets.iloc[-1] - 1) * 100

    # Sharpe ratio
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252)

    # Plot equity curve
    plt.figure(figsize=(12,6))
    plt.plot(cum_rets.index, cum_rets, color='blue', linewidth=2)
    plt.title(f'Equity Curve | Total Return: {total_ret:.2f}% | Sharpe Ratio: {sharpe:.2f}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.show()

#=========================================================================#

def plot_drawdown(cum_rets):
    # DRAWDOWN CURVE
    # Drawdown calculation
    running_max = cum_rets.cummax()
    drawdown = (cum_rets / running_max) - 1
    max_dd = drawdown.min() * 100

    # Plot drawdown curve
    plt.figure(figsize=(12,6))
    plt.plot(drawdown.index, drawdown, color='red', linewidth=2)
    plt.fill_between(x=drawdown.index,
                    y1=drawdown,
                    y2=0,
                    color='red',
                    alpha=0.1)
    plt.title(f'rawdown Curve | Max Drawdown: {max_dd:.2f}%')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True)
    plt.show()