import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

import {
  type Alert,
  type AlertCondition,
  type AlertType,
  useAlertsStore,
} from "../../store/alerts";

const ALERT_TYPES: Array<{ label: string; value: AlertType }> = [
  { label: "Price", value: "price" },
  { label: "OI", value: "oi" },
  { label: "PCR", value: "pcr" },
];

const CONDITIONS: Array<{ label: string; value: AlertCondition }> = [
  { label: "Above", value: "above" },
  { label: "Below", value: "below" },
  { label: "Crosses Up", value: "crosses_up" },
  { label: "Crosses Down", value: "crosses_down" },
];

const numberFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 2,
  minimumFractionDigits: 2,
});

const dateFormatter = new Intl.DateTimeFormat("en-IN", {
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  month: "short",
});

export function Alerts() {
  const alerts = useAlertsStore((state) => state.alerts);
  const setAlerts = useAlertsStore((state) => state.setAlerts);
  const removeAlert = useAlertsStore((state) => state.removeAlert);
  const [symbol, setSymbol] = useState("NIFTY");
  const [alertType, setAlertType] = useState<AlertType>("price");
  const [condition, setCondition] = useState<AlertCondition>("above");
  const [threshold, setThreshold] = useState("");
  const [message, setMessage] = useState("Loading alerts...");
  const [isSaving, setIsSaving] = useState(false);

  const sortedAlerts = useMemo(
    () => [...alerts].sort((left, right) => Number(right.active) - Number(left.active)),
    [alerts],
  );

  const loadAlerts = useCallback(async () => {
    try {
      const nextAlerts = await invoke<Alert[]>("get_alerts");
      setAlerts(nextAlerts);
      setMessage("");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }, [setAlerts]);

  useEffect(() => {
    loadAlerts();

    const unlisten = listen("alert_triggered", () => {
      loadAlerts();
    });

    return () => {
      unlisten.then((dispose) => dispose()).catch(() => undefined);
    };
  }, [loadAlerts]);

  const createAlert = async () => {
    const parsedThreshold = Number(threshold);
    if (!Number.isFinite(parsedThreshold) || parsedThreshold < 0) {
      setMessage("Enter a non-negative threshold");
      return;
    }

    setIsSaving(true);
    setMessage("");

    try {
      await invoke<Alert>("set_alert", {
        alertType,
        condition,
        symbol,
        threshold: parsedThreshold,
      });
      setThreshold("");
      await loadAlerts();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setIsSaving(false);
    }
  };

  const deleteAlert = async (id: string) => {
    removeAlert(id);
    try {
      await invoke("delete_alert", { id });
      await loadAlerts();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
      await loadAlerts();
    }
  };

  return (
    <section style={sectionStyle}>
      <div style={headerRowStyle}>
        <h2 style={titleStyle}>Alerts</h2>
        <span style={hintStyle}>{alerts.filter((alert) => alert.active).length} active</span>
      </div>

      <div style={formGridStyle}>
        <input
          aria-label="Alert symbol"
          onChange={(event) => setSymbol(event.target.value)}
          placeholder="Symbol"
          style={controlStyle}
          value={symbol}
        />
        <select
          aria-label="Alert type"
          onChange={(event) => setAlertType(event.target.value as AlertType)}
          style={controlStyle}
          value={alertType}
        >
          {ALERT_TYPES.map((item) => (
            <option key={item.value} value={item.value}>
              {item.label}
            </option>
          ))}
        </select>
        <select
          aria-label="Alert condition"
          onChange={(event) => setCondition(event.target.value as AlertCondition)}
          style={controlStyle}
          value={condition}
        >
          {CONDITIONS.map((item) => (
            <option key={item.value} value={item.value}>
              {item.label}
            </option>
          ))}
        </select>
        <input
          aria-label="Alert threshold"
          min={0}
          onChange={(event) => setThreshold(event.target.value)}
          placeholder="Threshold"
          style={controlStyle}
          type="number"
          value={threshold}
        />
        <button
          disabled={isSaving}
          onClick={createAlert}
          style={buttonStyle(!isSaving)}
          type="button"
        >
          Create Alert
        </button>
      </div>

      {message ? <div style={messageStyle}>{message}</div> : null}

      <div style={{ border: "1px solid #2a2e39", overflowX: "auto" }}>
        <table style={tableStyle}>
          <thead>
            <tr>
              {["Symbol", "Type", "Condition", "Threshold", "Status", "Created", "Actions"].map(
                (heading) => (
                  <th key={heading} style={headerCellStyle}>
                    {heading}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {sortedAlerts.length > 0 ? (
              sortedAlerts.map((alert) => (
                <tr key={alert.id}>
                  <td style={cellStyle}>{alert.symbol}</td>
                  <td style={cellStyle}>{labelForAlertType(alert.alert_type)}</td>
                  <td style={cellStyle}>{labelForCondition(alert.condition)}</td>
                  <td style={cellStyle}>{numberFormatter.format(alert.threshold)}</td>
                  <td style={cellStyle}>
                    {alert.active ? (
                      <span style={activeStatusStyle}>Active</span>
                    ) : (
                      <span style={triggeredStatusStyle}>
                        Triggered
                        {alert.triggered_at ? ` ${formatTime(alert.triggered_at)}` : ""}
                      </span>
                    )}
                  </td>
                  <td style={cellStyle}>{formatTime(alert.created_at)}</td>
                  <td style={cellStyle}>
                    <button
                      onClick={() => deleteAlert(alert.id)}
                      style={deleteButtonStyle}
                      type="button"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={7} style={{ ...cellStyle, color: "#8a94a6" }}>
                  No alerts yet
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function formatTime(timestamp: number) {
  return dateFormatter.format(new Date(timestamp));
}

function labelForAlertType(alertType: AlertType) {
  return alertType === "price" ? "Price" : alertType === "oi" ? "OI" : "PCR";
}

function labelForCondition(condition: AlertCondition) {
  return CONDITIONS.find((item) => item.value === condition)?.label ?? condition;
}

function buttonStyle(enabled: boolean): React.CSSProperties {
  return {
    background: enabled ? "#1f2937" : "#111111",
    border: "1px solid #333333",
    borderRadius: 6,
    color: enabled ? "#ffffff" : "#5f6673",
    cursor: enabled ? "pointer" : "not-allowed",
    fontFamily: "monospace",
    fontSize: 12,
    height: 34,
    padding: "0 12px",
  };
}

const sectionStyle = {
  borderTop: "1px solid #1f2933",
  color: "#d1d4dc",
  display: "flex",
  flexDirection: "column",
  gap: 14,
  minWidth: 0,
  paddingTop: 20,
  width: "100%",
} satisfies React.CSSProperties;

const headerRowStyle = {
  alignItems: "center",
  display: "flex",
  gap: 12,
  justifyContent: "space-between",
} satisfies React.CSSProperties;

const titleStyle = {
  color: "#f8fafc",
  fontSize: 18,
  letterSpacing: 0,
  margin: 0,
} satisfies React.CSSProperties;

const hintStyle = {
  color: "#8a94a6",
  fontSize: 12,
} satisfies React.CSSProperties;

const formGridStyle = {
  display: "grid",
  gap: 10,
  gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
} satisfies React.CSSProperties;

const controlStyle = {
  background: "#101010",
  border: "1px solid #262626",
  borderRadius: 6,
  boxSizing: "border-box",
  color: "#ffffff",
  fontFamily: "monospace",
  fontSize: 12,
  height: 34,
  padding: "0 10px",
  width: "100%",
} satisfies React.CSSProperties;

const messageStyle = {
  color: "#facc15",
  fontSize: 12,
} satisfies React.CSSProperties;

const tableStyle = {
  borderCollapse: "collapse",
  fontSize: 12,
  minWidth: 900,
  tableLayout: "fixed",
  width: "100%",
} satisfies React.CSSProperties;

const headerCellStyle = {
  background: "#111827",
  borderBottom: "1px solid #2a2e39",
  color: "#9ca3af",
  fontWeight: 700,
  padding: "9px 8px",
  textAlign: "left",
} satisfies React.CSSProperties;

const cellStyle = {
  borderBottom: "1px solid #1f2933",
  padding: "8px",
  verticalAlign: "middle",
} satisfies React.CSSProperties;

const activeStatusStyle = {
  color: "#22c55e",
  fontWeight: 700,
} satisfies React.CSSProperties;

const triggeredStatusStyle = {
  color: "#8a94a6",
  fontWeight: 700,
} satisfies React.CSSProperties;

const deleteButtonStyle = {
  ...buttonStyle(true),
  background: "#1a0a0a",
  borderColor: "#3a1a1a",
  color: "#ef5350",
  height: 28,
  padding: "0 10px",
} satisfies React.CSSProperties;
