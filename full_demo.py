"""Full pipeline: optimize all strategies on a stock basket, then paper-trade."""
import asyncio
from datetime import datetime, timedelta

from src.data.yfinance_provider import YFinanceProvider
import src.strategies
from src.strategies.base import StrategyConfig
from src.strategies.registry import StrategyRegistry
from src.execution.alpaca_broker import AlpacaBroker
from src.optimization.selector import StrategySelector
from src.optimization.genetic import GeneticOptimizer
from src.optimization.persistence import save_config
from src.optimization.objectives import sharpe_ratio


SYMBOLS = ["AAPL", "MSFT", "SPY", "TSLA"]
PARAM_GRID = {
    "entry_threshold": [0.1, 0.3, 0.5, 0.7, 1.0],
    "exit_threshold": [0.05, 0.1, 0.2, 0.3, 0.5],
    "stop_loss_pct": [0.005, 0.01, 0.02, 0.03, 0.05],
    "take_profit_pct": [0.01, 0.02, 0.05, 0.10],
    "max_position_pct": [0.05, 0.10, 0.20],
}


async def main():
    provider = YFinanceProvider()
    end = datetime.now()
    start = end - timedelta(days=90)
    
    registry = src.strategies.registry
    optimizer = GeneticOptimizer(objective_fn=sharpe_ratio)
    selector = StrategySelector(registry=registry, optimizer=optimizer)
    
    all_results = {}
    
    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"📊 {symbol}: pulling 90 days of 1h data...")
        mdf = await provider.fetch_bars(symbol, start=start, end=end, timeframe="1h")
        data = mdf.df
        print(f"   {len(data)} bars")
        
        print(f"🧬 {symbol}: genetic optimization across all strategies...")
        result = selector.select_best(
            strategy_names=["momentum", "mean_reversion", "breakout"],
            data=data,
            param_grid=PARAM_GRID,
            n_iter=200,
        )
        
        best = result.best_config
        print(f"   ✅ Best: {result.best_strategy}")
        print(f"      Sharpe: {result.best_score:.4f}")
        print(f"      entry: {best.entry_threshold:.3f}  exit: {best.exit_threshold:.3f}")
        print(f"      sl: {best.stop_loss_pct:.3f}  tp: {best.take_profit_pct:.3f}")
        
        path = f"/home/team/shared/engine/configs/{symbol}_{result.best_strategy}.json"
        save_config(best, path)
        print(f"      Saved → {path}")
        
        all_results[symbol] = result
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 OPTIMIZATION SUMMARY")
    print(f"{'Symbol':<8} {'Strategy':<18} {'Sharpe':>8}")
    print("-" * 40)
    for sym, r in all_results.items():
        print(f"{sym:<8} {r.best_strategy:<18} {r.best_score:>8.4f}")
    
    # Pick best overall
    best_symbol = max(all_results, key=lambda s: all_results[s].best_score)
    best_result = all_results[best_symbol]
    
    print(f"\n🚀 Starting paper trading: {best_result.best_strategy} on {best_symbol}")
    broker = AlpacaBroker()
    account = await broker.get_account()
    print(f"   Account: ${float(account.get('equity', 0)):,.2f}")
    
    from src.execution.live_runner import LiveTradingRunner
    
    if best_result.best_strategy == "momentum":
        from src.strategies.momentum import MomentumStrategy as StratCls
    elif best_result.best_strategy == "mean_reversion":
        from src.strategies.mean_reversion import MeanReversionStrategy as StratCls
    else:
        from src.strategies.breakout import BreakoutStrategy as StratCls
    
    runner = LiveTradingRunner(
        symbols=[best_symbol],
        strategy_cls=StratCls,
        broker=broker,
        data_provider=YFinanceProvider(),
        config=best_result.best_config,
        check_interval_seconds=30.0,
        confidence_threshold=0.3,
        max_positions=3,
        close_on_shutdown=False,
    )
    
    print(f"   Running 3 check cycles...")
    for i in range(3):
        print(f"\n⏱  Tick {i+1}/3...")
        try:
            await runner._tick()
            positions = await broker.get_positions()
            if positions:
                for p in positions:
                    print(f"   📈 {p.get('symbol')}: {p.get('qty')} shares @ ${float(p.get('market_value', 0)):,.2f} | "
                          f"P&L: ${float(p.get('unrealized_pl', 0)):,.2f}")
            else:
                print(f"   No open positions")
            account = await broker.get_account()
            print(f"   Equity: ${float(account.get('equity', 0)):,.2f}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    await broker.close()
    print(f"\n✅ Done! Configs saved to /home/team/shared/engine/configs/")


if __name__ == "__main__":
    asyncio.run(main())
