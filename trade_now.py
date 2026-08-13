"""Force a paper trade — mean reversion on AAPL, bypass market check."""
import asyncio
from datetime import datetime, timedelta

from src.data.yfinance_provider import YFinanceProvider
import src.strategies
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.base import StrategyConfig
from src.execution.alpaca_broker import AlpacaBroker


async def main():
    symbol = "AAPL"
    
    # Config tuned for trading — very sensitive
    config = StrategyConfig(
        entry_threshold=0.5,      # z-score threshold to enter (low = more signals)
        exit_threshold=0.1,       # z-score to exit
        stop_loss_pct=0.03,
        take_profit_pct=0.03,
        max_position_pct=0.10,
        extra={"lookback": 20, "std_dev_multiplier": 2.0},
    )
    
    # Pull recent data
    provider = YFinanceProvider()
    end = datetime.now()
    start = end - timedelta(days=5)
    mdf = await provider.fetch_bars(symbol, start=start, end=end, timeframe="1min")
    data = mdf.df
    print(f"📊 {len(data)} bars | last close: ${data['close'].iloc[-1]:.2f}")
    
    # See what signals the strategy generates
    strategy = MeanReversionStrategy(config=config)
    signals = strategy.generate_signals(data)
    
    buys = [s for s in signals if s.signal_type.value == "BUY"]
    sells = [s for s in signals if s.signal_type.value == "SELL"]
    print(f"\n📡 Mean Reversion signals:")
    print(f"   BUY:  {len(buys)}")
    print(f"   SELL: {len(sells)}")
    
    # Show last few signals with decent confidence
    confident = [s for s in signals[-50:] if s.confidence > 0.3]
    for s in confident[-10:]:
        print(f"   {s.timestamp} | {s.signal_type.value} | conf={s.confidence:.3f}")
    
    if not confident:
        print(f"   No signals above 0.3 confidence in last 50 bars")
        print(f"   Latest signal: {signals[-1].signal_type.value} | conf={signals[-1].confidence:.3f}")
    
    # Now try to trade — bypass market hours check
    print(f"\n🔗 Alpaca paper trading...")
    broker = AlpacaBroker()
    account = await broker.get_account()
    print(f"   Equity: ${float(account.get('equity', 0)):,.2f}")
    print(f"   Market open: {await broker.is_market_open()}")
    
    # Get the latest signal
    latest = signals[-1]
    current_price = float(data['close'].iloc[-1])
    
    print(f"\n🚀 Attempting trade based on latest signal:")
    print(f"   {latest.signal_type.value} {symbol} | conf={latest.confidence:.3f} | price=${current_price:.2f}")
    
    if latest.signal_type.value == "BUY" and latest.confidence > 0.2:
        from src.execution.broker import Order, OrderSide, OrderType
        
        max_val = float(account.get('equity', 100000)) * config.max_position_pct
        qty = max_val / current_price
        order = Order(symbol=symbol, side=OrderSide.BUY, quantity=qty, order_type=OrderType.MARKET)
        
        print(f"   Placing MARKET BUY: {qty:.2f} shares = ~${max_val:,.2f}")
        result = await broker.place_order(order)
        print(f"   ✅ Order: {result.order_id} | status={result.status} | filled={result.filled_quantity} @ ${result.filled_avg_price}")
        
        await asyncio.sleep(2)
        positions = await broker.get_positions()
        if positions:
            for p in positions:
                print(f"   📈 Active: {p.get('symbol')} {p.get('qty')} shares | "
                      f"P&L: ${float(p.get('unrealized_pl', 0)):,.2f}")
        
        # Close it immediately
        if positions:
            print(f"\n   🔄 Closing position...")
            sell_order = Order(symbol=symbol, side=OrderSide.SELL, quantity=qty, order_type=OrderType.MARKET)
            sell_result = await broker.place_order(sell_order)
            print(f"   ✅ Closed: {sell_result.order_id} | status={sell_result.status}")
    else:
        print(f"   ⚠️  Signal too weak or wrong type — not trading")
    
    # Final account state
    account = await broker.get_account()
    positions = await broker.get_positions()
    print(f"\n📋 Final: Equity=${float(account.get('equity', 0)):,.2f} | Positions={len(positions)}")
    await broker.close()


if __name__ == "__main__":
    asyncio.run(main())
