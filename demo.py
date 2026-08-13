"""Demo: optimize a strategy on real data, then paper-trade live."""
import asyncio
import os

from src.data.yfinance_provider import YFinanceProvider
from src.data.provider import DataProvider
import src.strategies  # registers all strategies in StrategyRegistry
from src.strategies.momentum import MomentumStrategy
from src.strategies.base import StrategyConfig
from src.execution.alpaca_broker import AlpacaBroker
from src.execution.live_runner import LiveTradingRunner
from src.optimization.runner import run_optimization


async def main():
    symbol = "AAPL"
    
    # ── Step 1: Pull real historical data ──────────────────────────
    from datetime import datetime, timedelta
    
    print(f"📊 Pulling 60 days of historical data for {symbol}...")
    provider = YFinanceProvider()
    end = datetime.now()
    start = end - timedelta(days=60)
    mdf = await provider.fetch_bars(symbol, start=start, end=end, timeframe="1h")
    data = mdf.df
    print(f"   Got {len(data)} bars ({data.index[0]} → {data.index[-1]})")
    
    # ── Step 2: Run the AI optimizer ───────────────────────────────
    print(f"\n🧠 Optimizing MomentumStrategy for {symbol}...")
    result = run_optimization(
        strategy_name="momentum",
        data=data,
        method="grid",
    )
    
    best = result.best_config
    print(f"\n   ✅ Best config found:")
    print(f"      entry_threshold:  {best.entry_threshold:.3f}")
    print(f"      exit_threshold:   {best.exit_threshold:.3f}")
    print(f"      stop_loss_pct:    {best.stop_loss_pct:.3f}")
    print(f"      take_profit_pct:  {best.take_profit_pct:.3f}")
    print(f"      max_position_pct: {best.max_position_pct:.3f}")
    print(f"      Sharpe score:     {result.best_score:.4f}")
    
    # ── Step 3: Start live paper trading ───────────────────────────
    print(f"\n🚀 Starting live paper trading for {symbol}...")
    
    broker = AlpacaBroker()
    account = await broker.get_account()
    print(f"   Account equity: ${float(account.get('equity', 0)):,.2f}")
    
    runner = LiveTradingRunner(
        symbols=[symbol],
        strategy_cls=MomentumStrategy,
        broker=broker,
        data_provider=YFinanceProvider(),
        config=best,
        check_interval_seconds=30.0,
        confidence_threshold=0.3,
        max_positions=3,
        close_on_shutdown=False,  # paper trading — safe to leave open
    )
    
    print(f"   Strategy: MomentumStrategy (AI-optimized)")
    print(f"   Polling every 30s | Confidence threshold: 0.3")
    print(f"   Press Ctrl+C to stop\n")
    
    try:
        await runner.run()
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
    finally:
        await broker.close()


if __name__ == "__main__":
    asyncio.run(main())
