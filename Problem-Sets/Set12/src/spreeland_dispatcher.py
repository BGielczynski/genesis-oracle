"""Spreeland Dispatcher prototype for Problem Set 12.

The script is intentionally runnable without real Spreeland infrastructure. It
demonstrates the requested protocol boundaries with deterministic mock data:

- MCP-style tool discovery for bridge infrastructure.
- A2A-style agent cards and task messages for weather and supplier agents.
- UCP/AP2-style user approval, mandates, and receipt verification.
- AG-UI-style event streaming and an A2UI delivery status card payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs without adding a dotenv dependency."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def utc_now() -> str:
    fixed_time = os.getenv("SPREELAND_FIXED_TIME")
    if fixed_time:
        return fixed_time
    return datetime(2026, 7, 18, 7, 0, tzinfo=timezone.utc).isoformat(timespec="seconds")


def stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ToolDescriptor:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class BridgeStatus:
    bridge_id: str
    name: str
    status: str
    capacity_tons: float
    eta_open: str | None


@dataclass(frozen=True)
class AgentCard:
    agent_id: str
    name: str
    protocol: str
    endpoint: str
    capabilities: list[str]


@dataclass(frozen=True)
class WeatherAssessment:
    route_id: str
    risk_level: str
    river_level_trend: str
    recommendation: str


@dataclass(frozen=True)
class SupplierOffer:
    supplier_agent_id: str
    item: str
    quantity_tons: float
    price_eur: float
    valid_until: str


@dataclass(frozen=True)
class OwnerApproval:
    owner_id: str
    approved: bool
    approval_token: str
    constraints: dict[str, Any]


@dataclass(frozen=True)
class Mandate:
    mandate_id: str
    mandate_type: str
    subject: dict[str, Any]
    issued_at: str
    signature: str


@dataclass(frozen=True)
class FulfillmentReceipt:
    receipt_id: str
    supplier_agent_id: str
    mandate_id: str
    status: str
    verification_hash: str


class MockBridgeMCPClient:
    """Small local stand-in for an MCP bridge-status tool server."""

    def discover_tools(self) -> list[ToolDescriptor]:
        return [
            ToolDescriptor(
                name="get_bridge_status",
                description="Return current bridge status and freight capacity.",
                input_schema={"type": "object", "properties": {"region": {"type": "string"}}},
            ),
            ToolDescriptor(
                name="watch_bridge_capacity",
                description="Stream bridge capacity changes for a selected route.",
                input_schema={"type": "object", "properties": {"route_id": {"type": "string"}}},
            ),
        ]

    def get_bridge_status(self, region: str) -> list[BridgeStatus]:
        if region.lower() != "spreeland":
            raise ValueError(f"Unsupported region: {region}")
        return [
            BridgeStatus("BR-17", "Leipe North Bridge", "maintenance", 0.0, "2026-07-18T18:00:00+02:00"),
            BridgeStatus("BR-22", "Burg South Freight Bridge", "open", 7.5, None),
            BridgeStatus("BR-31", "Cottbus East Service Bridge", "restricted", 3.0, None),
        ]


def discover_weather_agent() -> AgentCard:
    return AgentCard(
        agent_id="weather-predictor-spreewald",
        name="Weather Predictor Agent",
        protocol="A2A",
        endpoint="https://agents.example.invalid/spreewald-weather",
        capabilities=["river-level-forecast", "route-risk-assessment"],
    )


def request_weather_assessment(agent: AgentCard, route_id: str, bridges: list[BridgeStatus]) -> WeatherAssessment:
    blocked = [bridge for bridge in bridges if bridge.status != "open"]
    risk = "medium" if blocked else "low"
    return WeatherAssessment(
        route_id=route_id,
        risk_level=risk,
        river_level_trend="rising",
        recommendation=(
            "Use the southern freight bridge and avoid low-clearance river roads after 17:00."
            if risk == "medium"
            else "Primary route is acceptable."
        ),
    )


def discover_supplier_agent() -> AgentCard:
    return AgentCard(
        agent_id="supplier-agent-gurkenhof-lehde",
        name="Gurkenhof Lehde Supplier Agent",
        protocol="A2A",
        endpoint="https://agents.example.invalid/gurkenhof-lehde",
        capabilities=["quote-produce", "reserve-inventory", "issue-fulfillment-receipt"],
    )


def negotiate_supplier_order(agent: AgentCard, item: str, quantity_tons: float) -> SupplierOffer:
    base_price_per_ton = 1820.0
    return SupplierOffer(
        supplier_agent_id=agent.agent_id,
        item=item,
        quantity_tons=quantity_tons,
        price_eur=round(quantity_tons * base_price_per_ton, 2),
        valid_until="2026-07-18T12:00:00+02:00",
    )


def request_owner_approval(offer: SupplierOffer) -> OwnerApproval:
    owner_id = os.getenv("SPREELAND_OWNER_ID", "owner-spreeland-farmers-coop")
    max_budget = float(os.getenv("SPREELAND_MAX_BUDGET_EUR", "4000"))
    approved = offer.price_eur <= max_budget
    payload = {"owner_id": owner_id, "offer": asdict(offer), "max_budget_eur": max_budget}
    return OwnerApproval(
        owner_id=owner_id,
        approved=approved,
        approval_token=stable_hash(payload),
        constraints={"max_budget_eur": max_budget, "max_quantity_tons": 2.0},
    )


def create_ap2_mandates(offer: SupplierOffer, approval: OwnerApproval) -> tuple[Mandate, Mandate]:
    intent_subject = {
        "intent": "purchase harvest for Cottbus delivery",
        "item": offer.item,
        "quantity_tons": offer.quantity_tons,
        "owner_id": approval.owner_id,
        "constraints": approval.constraints,
    }
    intent_mandate = Mandate(
        mandate_id="intent-2t-gherkins-cottbus",
        mandate_type="IntentMandate",
        subject=intent_subject,
        issued_at=utc_now(),
        signature=stable_hash({"type": "intent", "subject": intent_subject, "approval": approval.approval_token}),
    )

    cart_subject = {
        "supplier_agent_id": offer.supplier_agent_id,
        "item": offer.item,
        "quantity_tons": offer.quantity_tons,
        "price_eur": offer.price_eur,
        "intent_mandate_id": intent_mandate.mandate_id,
    }
    cart_mandate = Mandate(
        mandate_id="cart-gherkins-3640eur",
        mandate_type="CartMandate",
        subject=cart_subject,
        issued_at=utc_now(),
        signature=stable_hash({"type": "cart", "subject": cart_subject, "approval": approval.approval_token}),
    )
    return intent_mandate, cart_mandate


def verify_fulfillment_receipt(offer: SupplierOffer, cart_mandate: Mandate) -> FulfillmentReceipt:
    evidence = {"offer": asdict(offer), "cart_mandate": asdict(cart_mandate), "status": "reserved"}
    return FulfillmentReceipt(
        receipt_id="receipt-gherkins-20260718-001",
        supplier_agent_id=offer.supplier_agent_id,
        mandate_id=cart_mandate.mandate_id,
        status="reserved",
        verification_hash=stable_hash(evidence),
    )


def build_delivery_status_card(
    bridges: list[BridgeStatus],
    weather: WeatherAssessment,
    offer: SupplierOffer,
    approval: OwnerApproval,
    intent_mandate: Mandate,
    cart_mandate: Mandate,
    receipt: FulfillmentReceipt,
) -> dict[str, Any]:
    open_bridges = [bridge for bridge in bridges if bridge.status == "open"]
    selected_route_id = "SPREE-R2" if open_bridges else "HOLD"
    return {
        "schema": "a2ui.delivery_status_card.v1",
        "component": "DeliveryStatusCard",
        "title": "Spreeland Logistics Sync",
        "subtitle": "Burg/Spreewald to Cottbus harvest delivery",
        "status": "rerouted" if selected_route_id != "HOLD" else "blocked",
        "severity": "warning" if weather.risk_level != "low" else "info",
        "route": {
            "origin": "Burg/Spreewald Cooperative Hub",
            "destination": "Cottbus Wholesale Market",
            "selected_route_id": selected_route_id,
            "estimated_arrival": "2026-07-18T15:40:00+02:00" if selected_route_id != "HOLD" else None,
        },
        "bridges": [asdict(bridge) for bridge in bridges[:2]],
        "weather": {
            "agent_id": "weather-predictor-spreewald",
            **asdict(weather),
        },
        "fulfillment": {
            "supplier_agent_id": offer.supplier_agent_id,
            "item": offer.item,
            "quantity_tons": offer.quantity_tons,
            "offer_price_eur": offer.price_eur,
            "approval_status": "approved" if approval.approved else "rejected",
            "payment_protocol": "AP2",
            "commerce_context_protocol": "UCP",
        },
        "dispatcher": {
            "agent_id": "Spreeland_Dispatcher",
            "next_action": "Dispatch truck convoy via SPREE-R2 and monitor BR-22 capacity every 15 minutes.",
            "stream_channel": "ag-ui-events",
        },
        "audit": {
            "decision_id": "sync-20260718-spree-r2",
            "intent_mandate_id": intent_mandate.mandate_id,
            "cart_mandate_id": cart_mandate.mandate_id,
            "receipt_id": receipt.receipt_id,
            "verification_hash": receipt.verification_hash,
        },
        "actions": [
            {"id": "approve-dispatch", "label": "Approve Dispatch", "kind": "primary"},
            {"id": "request-weather-refresh", "label": "Refresh Weather Risk", "kind": "secondary"},
        ],
    }


def build_adk_dispatcher() -> tuple[Any | None, str]:
    """Build a google-adk Agent when optional runtime packages are available."""
    try:
        from google.adk.agents import Agent
        from google.adk.tools.mcp_tool import McpToolset
        from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
        from mcp import StdioServerParameters
    except Exception as exc:  # pragma: no cover - depends on local optional deps.
        return None, f"ADK/MCP runtime not available, using mock mode: {exc}"

    command = os.getenv("SPREELAND_BRIDGE_MCP_COMMAND", "npx")
    args = os.getenv("SPREELAND_BRIDGE_MCP_ARGS", "-y @spreeland/bridge-mcp-server").split()
    api_key = os.getenv("SPREELAND_BRIDGE_API_KEY", "SPREE_2026_SECRET")
    infra_tools = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=command,
                args=args,
                env={"API_KEY": api_key},
            )
        )
    )
    agent = Agent(
        model=os.getenv("SPREELAND_DISPATCHER_MODEL", "gemini-3.5-flash"),
        name="Spreeland_Dispatcher",
        instruction=(
            "Coordinate Spreeland logistics. Check bridge status through MCP, "
            "consult weather and supplier agents through A2A, require owner "
            "authorization before commerce, and stream user-visible status "
            "events suitable for AG-UI/A2UI frontends."
        ),
        tools=[infra_tools],
    )
    return agent, "ADK dispatcher configured with MCP bridge toolset."


def run_dispatch() -> Iterable[dict[str, Any]]:
    bridge_client = MockBridgeMCPClient()

    yield {"event": "run.started", "message": "Starting Spreeland logistics synchronization."}

    tools = bridge_client.discover_tools()
    yield {
        "event": "mcp.tools.discovered",
        "message": "Discovered bridge infrastructure tools through MCP.",
        "data": [asdict(tool) for tool in tools],
    }

    bridges = bridge_client.get_bridge_status("spreeland")
    yield {
        "event": "mcp.tool.result",
        "message": "Loaded real-time bridge status from the infrastructure tool.",
        "data": [asdict(bridge) for bridge in bridges],
    }

    weather_agent = discover_weather_agent()
    yield {
        "event": "a2a.agent.discovered",
        "message": "Discovered external Weather-Predictor Agent through A2A agent card.",
        "data": asdict(weather_agent),
    }

    weather = request_weather_assessment(weather_agent, "SPREE-R2", bridges)
    yield {
        "event": "a2a.task.completed",
        "message": "Received route risk assessment from Weather-Predictor Agent.",
        "data": asdict(weather),
    }

    supplier_agent = discover_supplier_agent()
    offer = negotiate_supplier_order(supplier_agent, "Spreewald gherkins", 2.0)
    yield {
        "event": "a2a.negotiation.offer",
        "message": "Negotiated supplier offer for 2 tons of gherkins.",
        "data": {"supplier": asdict(supplier_agent), "offer": asdict(offer)},
    }

    approval = request_owner_approval(offer)
    if not approval.approved:
        yield {
            "event": "ucp.owner_approval.rejected",
            "message": "Owner policy rejected the order.",
            "data": asdict(approval),
        }
        return

    yield {
        "event": "ucp.owner_approval.approved",
        "message": "Owner authorization satisfies UCP policy constraints.",
        "data": asdict(approval),
    }

    intent_mandate, cart_mandate = create_ap2_mandates(offer, approval)
    yield {
        "event": "ap2.mandates.created",
        "message": "Created AP2-style intent and cart mandates for auditability.",
        "data": {"intent_mandate": asdict(intent_mandate), "cart_mandate": asdict(cart_mandate)},
    }

    receipt = verify_fulfillment_receipt(offer, cart_mandate)
    yield {
        "event": "ap2.receipt.verified",
        "message": "Verified supplier fulfillment receipt against the cart mandate.",
        "data": asdict(receipt),
    }

    card = build_delivery_status_card(bridges, weather, offer, approval, intent_mandate, cart_mandate, receipt)
    yield {
        "event": "a2ui.card.rendered",
        "message": "Rendered delivery status card payload for the dashboard.",
        "data": card,
    }

    yield {"event": "run.completed", "message": "Spreeland logistics synchronization completed."}


def write_card(path: Path) -> None:
    events = list(run_dispatch())
    card_event = next(event for event in events if event["event"] == "a2ui.card.rendered")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card_event["data"], indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Spreeland Dispatcher mock protocol flow.")
    parser.add_argument("--json", action="store_true", help="Print AG-UI-style events as JSON lines.")
    parser.add_argument(
        "--write-card",
        type=Path,
        help="Write the generated A2UI card JSON to the given path and exit.",
    )
    parser.add_argument("--show-adk-status", action="store_true", help="Check optional google-adk/MCP setup.")
    return parser.parse_args()


def main() -> int:
    load_env_file(ROOT / ".env")
    args = parse_args()

    if args.show_adk_status:
        _, status = build_adk_dispatcher()
        print(status)
        return 0

    if args.write_card:
        write_card(args.write_card)
        print(f"Wrote A2UI card to {args.write_card}")
        return 0

    for event in run_dispatch():
        if args.json:
            print(json.dumps(event, ensure_ascii=False))
        else:
            print(f"[{event['event']}] {event['message']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
