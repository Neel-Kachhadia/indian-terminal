import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

import { FilingsTab } from "./FilingsTab";
import { FinancialsTab } from "./FinancialsTab";
import { NewsTab } from "./NewsTab";
import { ValuationTab } from "./ValuationTab";
import { ContentViewer } from "./ContentViewer";

const tabs = [
  { label: "VALUATION", value: "valuation" },
  { label: "NEWS", value: "news" },
  { label: "FINANCIALS", value: "financials" },
  { label: "FILINGS", value: "filings" },
];

const keyframes = `
@keyframes equity-skeleton-pulse {
  0% { opacity: 0.35; }
  50% { opacity: 0.8; }
  100% { opacity: 0.35; }
}
@keyframes equity-live-pulse {
  0% { transform: scale(0.8); opacity: 0.7; }
  70% { transform: scale(1.7); opacity: 0; }
  100% { transform: scale(0.8); opacity: 0; }
}`;

export function classifySummaryError(error) {
  const text = error instanceof Error ? error.message : String(error);
  return /\(404\b/.test(text) ? "UNAVAILABLE" : "ERROR";
}

export function classifyValuationError(error) {
  const text = error instanceof Error ? error.message : String(error);
  return /\(404\b/.test(text) ? "INSUFFICIENT_DATA" : "ERROR";
}

export function EquityPanel({ symbol, processToken = null }) {
  const [activeTab, setActiveTab] = useState("valuation");
  const [collapsed, setCollapsed] = useState(false);
  const [data, setData] = useState(null);
  const [summaryStatus, setSummaryStatus] = useState("LOADING");
  const [summaryMessage, setSummaryMessage] = useState("");
  const [valuation, setValuation] = useState(null);
  const [valuationStatus, setValuationStatus] = useState("LOADING");
  const [valuationMessage, setValuationMessage] = useState("");
  const [viewer, setViewer] = useState(null);
  const [height, setHeight] = useState(280);
  const dragRef = useRef(null);

  const loading = summaryStatus === "LOADING";
  const message = summaryStatus === "ERROR" || summaryStatus === "UNAVAILABLE" ? summaryMessage : "";

  // Summary and valuation are independently sourced and independently fail --
  // a summary outage (e.g. no news/filings row) must never blank an otherwise
  // valid valuation, and vice versa. Each gets its own settled state.
  useEffect(() => {
    if (!symbol) {
      setData(null);
      setSummaryStatus("LOADING");
      return;
    }

    let active = true;
    const load = async (showSkeleton) => {
      if (showSkeleton) {
        setSummaryStatus("LOADING");
      }
      try {
        const result = await invoke("get_equity_summary", { symbol });
        if (!active) {
          return;
        }
        setData(result);
        setSummaryStatus("READY");
        setSummaryMessage("");
      } catch (error) {
        if (!active) {
          return;
        }
        setData(null);
        setSummaryStatus(classifySummaryError(error));
        setSummaryMessage(error instanceof Error ? error.message : String(error));
      }
    };

    load(true);
    const timer = window.setInterval(() => load(false), 180_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [symbol, processToken]);

  useEffect(() => {
    if (!symbol) {
      setValuation(null);
      setValuationStatus("LOADING");
      return;
    }

    let active = true;
    const load = async (showSkeleton) => {
      if (showSkeleton) {
        setValuationStatus("LOADING");
      }
      try {
        const result = await invoke("get_equity_valuation", { symbol });
        if (!active) {
          return;
        }
        setValuation(result);
        setValuationStatus(result?.status === "VALID" ? "READY" : "INSUFFICIENT_DATA");
        setValuationMessage("");
      } catch (error) {
        if (!active) {
          return;
        }
        setValuation(null);
        setValuationStatus(classifyValuationError(error));
        setValuationMessage(error instanceof Error ? error.message : String(error));
      }
    };

    load(true);
    const timer = window.setInterval(() => load(false), 180_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [symbol, processToken]);

  useEffect(() => {
    const move = (event) => {
      if (!dragRef.current) {
        return;
      }
      const nextHeight = dragRef.current.height + dragRef.current.y - event.clientY;
      setHeight(Math.max(180, Math.min(520, nextHeight)));
    };
    const up = () => {
      dragRef.current = null;
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
  }, []);

  const counts = useMemo(
    () => ({
      filings: data?.filings?.length ?? 0,
      news: data?.news?.length ?? 0,
      quarters: data?.financials?.quarters?.length ?? 0,
    }),
    [data],
  );

  if (!symbol) {
    return null;
  }

  if (collapsed) {
    return (
      <div style={collapsedStyle} onClick={() => setCollapsed(false)} role="button" tabIndex={0}>
        <style>{keyframes}</style>
        [▼ NEWS & FILINGS — {counts.news} new]
      </div>
    );
  }

  return (
    <section style={{ ...panelStyle, height }} aria-label={`${symbol} news and filings`}>
      <style>{keyframes}</style>
      <div
        aria-label="Resize equity panel"
        onPointerDown={(event) => {
          dragRef.current = { height, y: event.clientY };
          event.currentTarget.setPointerCapture?.(event.pointerId);
        }}
        style={resizeHandleStyle}
      />
      <div style={headerStyle}>
        <button onClick={() => setCollapsed(true)} style={collapseButtonStyle} type="button">
          ▲
        </button>
        <span style={titleStyle}>{data?.company_name ?? symbol}</span>
        <span style={countStyle}>{counts.news} news / {counts.filings} filings</span>
      </div>
      <div style={tabBarStyle}>
        {tabs.map((tab) => (
          <button
            key={tab.value}
            onClick={() => setActiveTab(tab.value)}
            style={tabButtonStyle(activeTab === tab.value)}
            type="button"
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div style={bodyStyle}>
        {activeTab === "valuation" ? (
          valuationStatus === "LOADING" ? (
            <Skeleton />
          ) : (
            <ValuationTab valuation={valuation ?? {}} status={valuationStatus} message={valuationMessage} symbol={symbol} />
          )
        ) : null}
        {activeTab !== "valuation" && loading ? <Skeleton /> : null}
        {activeTab !== "valuation" && !loading && message ? <div style={messageStyle}>{message}</div> : null}
        {activeTab === "news" && !loading && !message ? (
          <NewsTab news={data?.news ?? []} onItemClick={setViewer} symbol={symbol} />
        ) : null}
        {activeTab === "financials" && !loading && !message ? (
          <FinancialsTab financials={data?.financials ?? {}} symbol={symbol} />
        ) : null}
        {activeTab === "filings" && !loading && !message ? (
          <FilingsTab filings={data?.filings ?? []} onItemClick={setViewer} symbol={symbol} />
        ) : null}
      </div>
      {viewer ? <ContentViewer item={viewer} onClose={() => setViewer(null)} /> : null}
    </section>
  );
}

function Skeleton() {
  return (
    <div style={skeletonWrapStyle}>
      {[0, 1, 2, 3].map((item) => (
        <div key={item} style={skeletonRowStyle}>
          <span style={{ ...skeletonBarStyle, width: "18%" }} />
          <span style={{ ...skeletonBarStyle, width: "58%" }} />
          <span style={{ ...skeletonBarStyle, width: "22%" }} />
        </div>
      ))}
    </div>
  );
}

const panelStyle = {
  background: "#0d0d0d",
  borderTop: "1px solid #1a1a2e",
  boxSizing: "border-box",
  color: "#e8e8f0",
  display: "grid",
  gridTemplateRows: "6px 30px 34px minmax(0, 1fr)",
  minHeight: 180,
  minWidth: 0,
  overflow: "hidden",
  position: "relative",
};

const resizeHandleStyle = {
  background: "#141414",
  borderTop: "1px solid #242424",
  cursor: "ns-resize",
};

const headerStyle = {
  alignItems: "center",
  borderBottom: "1px solid #171717",
  display: "grid",
  gridTemplateColumns: "32px minmax(0, 1fr) auto",
  minWidth: 0,
  padding: "0 10px 0 6px",
};

const collapseButtonStyle = {
  background: "transparent",
  border: 0,
  color: "#666666",
  cursor: "pointer",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 11,
  height: 26,
};

const titleStyle = {
  color: "#cccccc",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 12,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
};

const countStyle = {
  color: "#666666",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 10,
  paddingLeft: 10,
  whiteSpace: "nowrap",
};

const tabBarStyle = {
  alignItems: "stretch",
  background: "#0d0d0d",
  borderBottom: "1px solid #171717",
  display: "grid",
  gridTemplateColumns: "repeat(4, minmax(0, 120px)) minmax(0, 1fr)",
};

function tabButtonStyle(active) {
  return {
    background: "#0d0d0d",
    border: 0,
    borderBottom: active ? "2px solid #00d4aa" : "2px solid transparent",
    color: active ? "#ffffff" : "#666666",
    cursor: "pointer",
    fontFamily: "Inter, system-ui, sans-serif",
    fontSize: 10,
    fontWeight: active ? 600 : 400,
    minWidth: 0,
  };
}

const bodyStyle = {
  minHeight: 0,
  overflowY: "auto",
  padding: 10,
};

const collapsedStyle = {
  alignItems: "center",
  background: "#0d0d0d",
  borderTop: "1px solid #1a1a2e",
  color: "#00d4aa",
  cursor: "pointer",
  display: "flex",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 11,
  height: 30,
  padding: "0 12px",
};

const messageStyle = {
  color: "#d4aa00",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 12,
};

const skeletonWrapStyle = {
  display: "grid",
  gap: 8,
};

const skeletonRowStyle = {
  background: "#111111",
  border: "1px solid #1f1f1f",
  display: "flex",
  gap: 8,
  minHeight: 42,
  padding: 10,
};

const skeletonBarStyle = {
  animation: "equity-skeleton-pulse 1.2s ease-in-out infinite",
  background: "#2a2a2a",
  display: "block",
  height: 10,
  marginTop: 6,
};
