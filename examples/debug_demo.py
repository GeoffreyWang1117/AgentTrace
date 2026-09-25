#!/usr/bin/env python3
"""
Interactive debugging demonstration for AgentTrace.

This script creates a multi-agent scenario with an injected bug,
then demonstrates how AgentTrace helps identify the root cause.

Run with:
    python examples/debug_demo.py

Then open http://localhost:8000 to use the debugging UI.
"""

import asyncio
import random
import time
from dataclasses import dataclass

from agenttrace import Tracer, trace_agent, trace_tool
from agenttrace.api.server import create_app, set_storage
from agenttrace.storage.memory import MemoryStorage
from agenttrace.query.interface import QueryInterface


# === Simulated Bug Scenario ===
# An e-commerce order processing system where:
# 1. OrderAgent receives customer order
# 2. InventoryAgent checks stock
# 3. PaymentAgent processes payment
# 4. ShippingAgent creates shipment
#
# Bug: Intermittent failures in PaymentAgent due to
# upstream InventoryAgent returning stale data


# Global state (simulating shared state issues)
_inventory_cache = {}
_cache_timestamp = 0


@trace_tool(tool_name="check_inventory_api")
def check_inventory_api(product_id: str) -> dict:
    """External inventory API call."""
    time.sleep(0.05)  # Simulate API latency

    # Simulate intermittent stale cache
    global _inventory_cache, _cache_timestamp
    current_time = time.time()

    # Bug: Cache doesn't properly invalidate
    if product_id in _inventory_cache and current_time - _cache_timestamp < 10:
        return _inventory_cache[product_id]

    # Fresh data
    stock = random.randint(0, 100)
    data = {
        "product_id": product_id,
        "stock": stock,
        "warehouse": "WH-001",
        "last_updated": current_time,
    }

    _inventory_cache[product_id] = data
    _cache_timestamp = current_time

    return data


@trace_tool(tool_name="process_payment_api")
def process_payment_api(amount: float, method: str) -> dict:
    """External payment API call."""
    time.sleep(0.1)

    # Simulate payment processing
    success = random.random() > 0.1  # 90% success rate

    if not success:
        raise RuntimeError("Payment gateway timeout")

    return {
        "transaction_id": f"TXN-{random.randint(10000, 99999)}",
        "amount": amount,
        "status": "completed",
    }


@trace_tool(tool_name="create_shipment_api")
def create_shipment_api(order_id: str, address: str) -> dict:
    """External shipping API call."""
    time.sleep(0.08)

    return {
        "shipment_id": f"SHIP-{random.randint(10000, 99999)}",
        "order_id": order_id,
        "estimated_delivery": "3-5 business days",
    }


@dataclass
class InventoryAgent:
    """Manages inventory checks."""

    agent_id: str = "inventory_agent"

    @trace_agent()
    def check_availability(self, product_id: str, quantity: int) -> dict:
        """Check if product is available in required quantity."""
        inventory = check_inventory_api(product_id)

        available = inventory["stock"] >= quantity
        return {
            "product_id": product_id,
            "requested": quantity,
            "available_stock": inventory["stock"],
            "is_available": available,
            "warehouse": inventory["warehouse"],
        }


@dataclass
class PaymentAgent:
    """Handles payment processing."""

    agent_id: str = "payment_agent"

    @trace_agent()
    def process_payment(self, order: dict, inventory_check: dict) -> dict:
        """Process payment for an order."""
        # Bug trigger: If inventory data is stale, payment amount might be wrong
        if not inventory_check["is_available"]:
            raise ValueError(
                f"Cannot process payment: product {inventory_check['product_id']} "
                f"not available (stock: {inventory_check['available_stock']}, "
                f"requested: {inventory_check['requested']})"
            )

        amount = order["unit_price"] * order["quantity"]

        try:
            result = process_payment_api(amount, order.get("payment_method", "card"))
            return {
                "order_id": order["order_id"],
                "payment_status": "success",
                **result,
            }
        except RuntimeError as e:
            return {
                "order_id": order["order_id"],
                "payment_status": "failed",
                "error": str(e),
            }


@dataclass
class ShippingAgent:
    """Manages shipping creation."""

    agent_id: str = "shipping_agent"

    @trace_agent()
    def create_shipment(self, order: dict, payment: dict) -> dict:
        """Create a shipment for a paid order."""
        if payment["payment_status"] != "success":
            raise ValueError(f"Cannot ship unpaid order: {payment.get('error', 'payment failed')}")

        shipment = create_shipment_api(
            order["order_id"],
            order["shipping_address"],
        )

        return {
            "order_id": order["order_id"],
            "shipment": shipment,
            "status": "shipped",
        }


@dataclass
class OrderAgent:
    """Coordinates order processing."""

    agent_id: str = "order_agent"
    inventory: InventoryAgent = None
    payment: PaymentAgent = None
    shipping: ShippingAgent = None

    def __post_init__(self):
        self.inventory = self.inventory or InventoryAgent()
        self.payment = self.payment or PaymentAgent()
        self.shipping = self.shipping or ShippingAgent()

    @trace_agent()
    def process_order(self, order: dict) -> dict:
        """Process a complete order through all stages."""
        results = {"order_id": order["order_id"], "stages": []}

        # Stage 1: Check inventory
        inventory_result = self.inventory.check_availability(
            order["product_id"],
            order["quantity"],
        )
        results["stages"].append({"stage": "inventory", "result": inventory_result})

        if not inventory_result["is_available"]:
            results["status"] = "failed"
            results["error"] = "Out of stock"
            return results

        # Stage 2: Process payment
        payment_result = self.payment.process_payment(order, inventory_result)
        results["stages"].append({"stage": "payment", "result": payment_result})

        if payment_result["payment_status"] != "success":
            results["status"] = "failed"
            results["error"] = payment_result.get("error", "Payment failed")
            return results

        # Stage 3: Create shipment
        try:
            shipping_result = self.shipping.create_shipment(order, payment_result)
            results["stages"].append({"stage": "shipping", "result": shipping_result})
            results["status"] = "completed"
        except ValueError as e:
            results["status"] = "failed"
            results["error"] = str(e)

        return results


def create_sample_order(order_num: int) -> dict:
    """Create a sample order."""
    return {
        "order_id": f"ORD-{order_num:04d}",
        "customer_id": f"CUST-{random.randint(100, 999)}",
        "product_id": f"PROD-{random.choice(['A', 'B', 'C'])}",
        "quantity": random.randint(1, 10),
        "unit_price": round(random.uniform(10, 100), 2),
        "payment_method": random.choice(["card", "paypal", "bank"]),
        "shipping_address": "123 Main St, City, Country",
    }


def simulate_orders(num_orders: int = 10) -> Tracer:
    """Simulate processing multiple orders."""
    print(f"\n{'='*60}")
    print(f"Simulating {num_orders} orders...")
    print(f"{'='*60}\n")

    tracer = Tracer(run_id=f"orders_{int(time.time())}")
    tracer.start()

    order_agent = OrderAgent()
    results = {"success": 0, "failed": 0}

    for i in range(num_orders):
        order = create_sample_order(i + 1)
        print(f"Processing {order['order_id']}...", end=" ")

        try:
            result = order_agent.process_order(order)
            if result["status"] == "completed":
                results["success"] += 1
                print("SUCCESS")
            else:
                results["failed"] += 1
                print(f"FAILED: {result.get('error', 'Unknown')}")
        except Exception as e:
            results["failed"] += 1
            print(f"ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {results['success']} succeeded, {results['failed']} failed")
    print(f"{'='*60}")

    # Analyze errors
    errors = tracer.find_errors()
    if errors:
        print(f"\n Found {len(errors)} errors in trace:")
        query = QueryInterface(tracer.graph)

        for i, error in enumerate(errors[:3], 1):
            print(f"\n  Error {i}:")
            explanation = query.explain_error(error.id)
            print(f"    Type: {explanation['error_info']['type']}")
            print(f"    Agent: {explanation['error_info']['agent']}")
            print(f"    Root causes: {len(explanation['root_causes'])}")

            if explanation["likely_causes"]:
                print(f"    Most likely cause:")
                likely = explanation["likely_causes"][0]
                print(f"      - Score: {likely['score']:.2f}")
                for reason in likely["reasons"][:2]:
                    print(f"      - {reason}")

    return tracer


def main():
    """Main entry point."""
    import sys
    import uvicorn

    # Create storage and tracer
    storage = MemoryStorage()
    set_storage(storage)

    # Simulate orders and save trace
    tracer = simulate_orders(15)
    storage.save_graph(tracer.graph)

    # Run more simulations for comparison
    print("\nRunning additional simulations for comparison...")

    for i in range(3):
        t = simulate_orders(5)
        storage.save_graph(t.graph)

    tracer.stop()

    # Start the server
    print(f"\n{'='*60}")
    print("Starting AgentTrace Debugging UI...")
    print("Open http://localhost:8000 in your browser")
    print("Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    # Create and run the app
    app = create_app(storage=storage)

    # Add static file serving for UI
    from fastapi.staticfiles import StaticFiles
    from pathlib import Path

    ui_path = Path(__file__).parent.parent / "ui"
    if ui_path.exists():
        app.mount("/ui", StaticFiles(directory=str(ui_path)), name="ui")

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
