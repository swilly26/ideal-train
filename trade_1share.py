"""Place a real paper trade via Alpaca."""
import asyncio
from src.execution.alpaca_broker import AlpacaBroker
from src.execution.broker import Order, OrderSide, OrderType


async def main():
    broker = AlpacaBroker()
    
    account = await broker.get_account()
    print(f"Equity: ${float(account.get('equity', 0)):,.2f}")
    print(f"Market open: {await broker.is_market_open()}")
    
    positions = await broker.get_positions()
    print(f"Open positions: {len(positions)}")
    for p in positions:
        print(f"  {p.get('symbol')} {p.get('qty')} shares | P&L: ${float(p.get('unrealized_pl', 0)):,.2f}")
    
    # Place a small AAPL buy
    symbol = "AAPL"
    qty = 1  # just 1 share to test
    
    print(f"\n📤 Placing MARKET BUY: {qty} share of {symbol}...")
    order = Order(symbol=symbol, side=OrderSide.BUY, quantity=qty, order_type=OrderType.MARKET)
    result = await broker.place_order(order)
    print(f"   Order ID: {result.order_id}")
    print(f"   Status: {result.status}")
    print(f"   Filled: {result.filled_quantity} @ ${result.filled_avg_price}")
    print(f"   Created: {result.created_at}")
    
    await asyncio.sleep(2)
    
    # Check positions
    positions = await broker.get_positions()
    print(f"\n📋 Positions after trade: {len(positions)}")
    for p in positions:
        print(f"   {p.get('symbol')} {p.get('qty')} shares | "
              f"Value: ${float(p.get('market_value', 0)):,.2f} | "
              f"P&L: ${float(p.get('unrealized_pl', 0)):,.2f}")
    
    # Now sell it back
    if positions:
        for p in positions:
            qty = float(p.get('qty', 0))
            sym = p.get('symbol')
            print(f"\n📤 Placing MARKET SELL: {qty} shares of {sym}...")
            sell = Order(symbol=sym, side=OrderSide.SELL, quantity=qty, order_type=OrderType.MARKET)
            sr = await broker.place_order(sell)
            print(f"   Order ID: {sr.order_id}")
            print(f"   Status: {sr.status}")
            print(f"   Filled: {sr.filled_quantity} @ ${sr.filled_avg_price}")
    
    await asyncio.sleep(1)
    positions = await broker.get_positions()
    account = await broker.get_account()
    print(f"\n✅ Final: Equity=${float(account.get('equity', 0)):,.2f} | Positions={len(positions)}")
    
    await broker.close()


if __name__ == "__main__":
    asyncio.run(main())
