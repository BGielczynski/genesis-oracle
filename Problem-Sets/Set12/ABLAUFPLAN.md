# Ablaufplan fuer Problem Set 12

## Zielbild

Abzugeben ist kein reines Essay, sondern ein kleines, konsistentes
Multi-Agenten-System-Konzept mit drei belastbaren Artefakten:

- ein PDF-Design-Dokument zur Protokollwahl je Teilaufgabe,
- ein Python-Skript mit MCP-Tool-Discovery und A2A-Verhandlungslogik,
- ein JSON-Beispiel fuer eine A2UI-Lieferstatuskarte.

Das Szenario ist die resiliente Logistikkoordination im Spreeland: Bauern,
Bruecken, Wetter, Einkauf und Live-Status muessen durch einen
`SpreelandDispatcher` orchestriert werden.

## Protokoll-Zuordnung

| Teilproblem | Primaeres Protokoll | Begruendung | Nachweis in der Abgabe |
| --- | --- | --- | --- |
| Bridge-Status aus PostgreSQL abrufen | MCP | Der Dispatcher braucht kontrollierten Zugriff auf externe Infrastruktur-Tools und Datenquellen. MCP kapselt die Datenbank als Toolserver statt direktem DB-Zugriff im Agenten. | Architekturdiagramm im PDF, MCP-Tool-Discovery im Python-Skript |
| Weather-Predictor-Agent finden und befragen | A2A | Ein spezialisierter Agent eines anderen Teams wird als eigenstaendiger Kommunikationspartner behandelt. A2A beschreibt Discovery, Capability Exchange, Anfrage und Antwort zwischen Agenten. | Sequenzdiagramm im PDF, Mock-Verhandlung im Python-Skript |
| Einkauf von 2 Tonnen Gurken autorisieren und verifizieren | AP2 + UCP | AP2 wird fuer autorisierte, nachvollziehbare Agent-Payments/Fulfillment genutzt. UCP wird als Kontext- und Policy-Schicht eingeplant: Besitzerfreigabe, Budget, Identitaet, Rollen und Compliance-Regeln. | Policy-Flow im PDF, Signatur-/Approval-Mock im Python-Skript |
| Live-Dashboard ohne eigenes React/Flutter bauen | A2UI + AG-UI | A2UI beschreibt die strukturierte UI-Nachricht bzw. Karte. AG-UI beschreibt den Streaming-Interaktionsfluss zwischen Agent und Benutzeroberflaeche. | `delivery_status_card.json`, Streaming-Ausgabe im Python-Skript |

Hinweis: Die genaue UCP-Definition sollte an den Vorlesungsfolien gespiegelt
werden. Im Plan wird UCP als User/Context/Policy-Schicht verwendet, weil die
Aufgabe explizit Autorisierung durch den Besitzer verlangt.

## Abgabe-Struktur

Empfohlene Dateien in `Problem-Sets/Set12`:

```text
Set12/
  README.md
  ABLAUFPLAN.md
  problemset.html
  docs/
    Set12_Design_Document.md
    Set12_Design_Document.pdf
  src/
    spreeland_dispatcher.py
  schemas/
    delivery_status_card.json
  assets/
    img/
      week12_logistics.png
```

## Arbeitsphasen

### Phase 1: Anforderungen fixieren

Ergebnis: Eine klare Liste der Systemrollen und Abgabekriterien.

Aufgaben:

1. Rollen definieren:
   - `SpreelandDispatcher` als zentraler Koordinator.
   - `BridgeMCPServer` als Infrastruktur-Datenquelle.
   - `WeatherPredictorAgent` als externer Expertenagent.
   - `SupplierAgent` als Lieferant fuer Gurken/Ernteprodukte.
   - `Owner/User` als autorisierende Person.
   - `Dashboard UI` als A2UI/AG-UI-Ausgabe.
2. Datenobjekte festlegen:
   - Brueckenstatus: `bridge_id`, `status`, `capacity_tons`, `eta_open`.
   - Wetterprognose: `route_id`, `risk_level`, `river_level_trend`.
   - Bestellung: `item`, `quantity_tons`, `price`, `approval_token`.
   - Lieferstatuskarte: Route, Agentenstatus, Risiko, naechste Aktion.
3. Abnahmekriterien aus dem Problem Set uebernehmen:
   - PDF erklaert die Protokollwahl fuer jede Teilaufgabe.
   - Python zeigt MCP-Discovery und A2A-Negotiation.
   - JSON rendert konzeptionell eine A2UI-Delivery-Status-Card.

### Phase 2: Design-Dokument erstellen

Ergebnis: `docs/Set12_Design_Document.md`, spaeter als PDF exportiert.

Inhalt:

1. Kurzbeschreibung des Szenarios und Zielsystems.
2. Architekturdiagramm:
   - User -> AG-UI -> SpreelandDispatcher.
   - Dispatcher -> MCP -> Bridge/PostgreSQL Toolserver.
   - Dispatcher -> A2A -> WeatherPredictorAgent.
   - Dispatcher -> A2A/AP2 -> SupplierAgent/Fulfillment.
   - Dispatcher -> A2UI -> Delivery Dashboard Card.
3. Protokollentscheidung pro Exercise-1-Bullet:
   - Warum MCP fuer Datenbankstatus.
   - Warum A2A fuer externe Spezialagenten.
   - Warum AP2/UCP fuer Einkauf, Freigabe und Auditierbarkeit.
   - Warum A2UI/AG-UI fuer dynamische UI ohne Custom-App.
4. Failure Modes:
   - MCP-Server nicht erreichbar.
   - Wetteragent liefert keine Antwort.
   - Besitzerfreigabe fehlt.
   - Bruecke wird waehrend der Route gesperrt.
5. Sicherheits- und Audit-Konzept:
   - Keine Secrets hardcoden.
   - Approval-Token nur aus Umgebung oder Mock-Provider.
   - Jede Kaufentscheidung mit `decision_id` und Hash/Signature protokollieren.

### Phase 3: Python-Prototyp bauen

Ergebnis: `src/spreeland_dispatcher.py`.

Mindestumfang:

1. Agent-Definition nach Aufgaben-Snippet:
   - `Agent(name="Spreeland_Dispatcher", ...)`.
   - MCP-Toolset fuer Bridge-Status.
2. MCP-Discovery demonstrieren:
   - Toolserver-Konfiguration aus Umgebungsvariablen oder Mock-Fallback.
   - Ausgabe der gefundenen Tools bzw. eines simulierten Toolkatalogs.
3. A2A-Verhandlung demonstrieren:
   - `discover_weather_agent()`.
   - `request_weather_assessment(route)`.
   - `negotiate_supplier_order(item, quantity_tons)`.
4. AP2/UCP-Mock einbauen:
   - `request_owner_approval(order)`.
   - `create_payment_authorization(order, approval)`.
   - `verify_fulfillment_receipt(receipt)`.
5. AG-UI-Streaming simulieren:
   - Schrittweise Statusmeldungen per Generator oder async Stream.
   - Beispiel: "checking bridges", "consulting weather agent", "awaiting owner approval", "rendering A2UI card".
6. Keine echte externe Infrastruktur voraussetzen:
   - Falls `google-adk` oder MCP-Server nicht installiert sind, soll ein Mock-Modus laufen.
   - Das ist wichtig, damit die Abgabe demonstrierbar bleibt.

### Phase 4: A2UI-JSON erstellen

Ergebnis: `schemas/delivery_status_card.json`.

Pflichtinhalte:

- Card-Typ, Titel und Status.
- Route von Spreeland/Burg nach Cottbus.
- Brueckenstatus mit mindestens zwei Bruecken.
- Wetterrisiko und Flusspegeltrend.
- Einkaufs-/Fulfillment-Status.
- Naechste Aktion des Dispatchers.
- Audit-Felder: `decision_id`, `approval_status`, `verification_hash`.

### Phase 5: Integration pruefen

Ergebnis: Alle Abgabedateien sind konsistent.

Checks:

1. `README.md` verweist auf die drei Abgabeartefakte.
2. Python-Skript startet im Mock-Modus ohne Secrets.
3. JSON ist valide.
4. Design-Dokument nennt alle sechs Protokolle: MCP, A2A, UCP, AP2, A2UI, AG-UI.
5. PDF ist aus dem Markdown erzeugt und enthaelt keine lokalen Pfade, die beim Dozenten nicht funktionieren.

## Empfohlene Reihenfolge

1. `schemas/delivery_status_card.json` zuerst bauen, weil es das Datenmodell klaert.
2. `src/spreeland_dispatcher.py` als Mock-Prototyp implementieren.
3. Aus dem funktionierenden Ablauf das Design-Dokument schreiben.
4. PDF exportieren.
5. Finalen README-Abschnitt mit Ausfuehrungs- und Abgabehinweisen ergaenzen.

## Definition of Done

Das Problem Set ist abgabereif, wenn folgende Punkte erfuellt sind:

- Das PDF erklaert jede Protokollentscheidung explizit und bezieht sich auf die vier Exercise-1-Fragen.
- Das Python-Skript zeigt sichtbar MCP-Tool-Discovery und A2A-Agentenkommunikation.
- Der Einkaufspfad enthaelt eine Besitzerfreigabe und einen verifizierbaren Receipt-/Hash-Schritt.
- Die A2UI-JSON-Datei ist syntaktisch gueltig und beschreibt eine konkrete Lieferstatuskarte.
- Der Streaming-Ablauf ist in der Konsole oder im Code nachvollziehbar.
- Die Abgabe funktioniert auch ohne echte Spreeland-Infrastruktur durch Mock-Daten.
