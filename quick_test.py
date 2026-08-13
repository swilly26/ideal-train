"""Quick live test — force trading with aggressive settings."""
import asyncio
from datetime import datetime, timedelta

from src.data.yfinance_provider import YFinanceProvider
import src.strategies
from src.strategies.momentum import MomentumStrategy
from src.strategies.base import StrategyConfig
from src.execution.alpaca_broker import AlpacaBroker
from src.execution.live_runner import LiveTradingRunner


async def main():
    symbol = "AAPL"
    
    # Aggressive config — low thresholds, wide stop/take-profit
    config = StrategyConfig(
        entry_threshold=0.1,     # low bar to enter
        exit_threshold=0.05,     # low bar to exit
        stop_loss_pct=0.05,      # 5% stop
        take_profit_pct=0.05,    # 5% take profit
        max_position_pct=0.10,   # 10% per position
    )
    
    print(f"🔧 Config: entry={config.entry_threshold}, exit={config.exit_threshold}, "
          f"sl={config.stop_loss_pct}, tp={config.take_profit_pct}")
    
    # Pull latest data to see what signals would generate
    provider = YFinanceProvider()
    end = datetime.now()
    start = end - timedelta(days=5)
    mdf = await provider.fetch_bars(symbol, start=start, end=end, timeframe="1min")
    data = mdf.df
    print(f"📊 {len(data)} 1-min bars for {symbol} ({data.index[0]} → {data.index[-1]})")
    print(f"   Last close: ${data['close'].iloc[-1]:.2f}")
    
    # Generate signals to see what the strategy thinks
    strategy = MomentumStrategy(config=config)
    signals = strategy.generate_signals(data)
    
    buys = [s for s in signals if s.signal_type.value == "BUY"]
    sells = [s for s in signals if s.signal_type.value == "SELL"]
    print(f"\n📡 Strategy signals from recent data:")
    print(f"   BUY signals:  {len(buys)}")
    print(f"   SELL signals: {len(sells)}")
    
    for s in signals[-5:]:
        print(f"   {s.timestamp} | {s.signal_type.value} | confidence={s.confidence:.3f}")
    
    # Now connect to Alpaca and run a tick
    print(f"\n🔗 Connecting to Alpaca paper...")
    broker = AlpacaBroker()
    account = await broker.get_account()
    print(f"   Equity: ${float(account.get('equity', 0)):,.2f}")
    market_open = await broker.is_market_open()
    print(f"   Market open: {market_open}")
    
    runner = LiveTradingRunner(
        symbols=[symbol],
        strategy_cls=MomentumStrategy,
        broker=broker,
        data_provider=YFinanceProvider(),
        config=config,
        check_interval_seconds=5.0,
        confidence_threshold=0.1,
        max_positions=3,
        close_on_shutdown=False,
    )
    
    # Override market check for demo
    print(f"\n🚀 Running 3 forced ticks (ignoring market hours)...")
    
    for i in range(3):
        print(f"\n--- Tick {i+1} ---")
        try:
            await runner._tick()
            positions = await broker.get_positions()
            if positions:
                for p in positions:
                    print(f"📈 POSITION: {p.get('symbol')} | {p.get('qty')} shares | "
                          f"Value: ${float(p.get('market_value', 0)):,.2f} | "
                          f"P&L: ${float(p.get('unrealized_pl', 0)):,.2f}")
            else:
                print(f"   No open positions")
            account = await broker.get_account()
            print(f"   Equity: ${float(account.get('equity', 0)):,.2f}")
        except Exception as e:
            print(f"   ❌ Error: {type(e).__name__}: {e}")
        
        await asyncio.sleep(2)
    
    # Final summary
    positions = await broker.get_positions()
    account = await broker.get_account()
    print(f"\n✅ FINAL:")
    print(f"   Positions: {len(positions)}")
    print(f"   Equity: ${float(account.get('equity', 0)):,.2f}")
    
    await broker.close()


if __name__ == "__main__":
    asyncio.run(main())
