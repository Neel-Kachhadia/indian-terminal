import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

import { ApexHeader } from "./components/layout/ApexHeader";
import { DetailTabs } from "./components/layout/DetailTabs";
import { NiftyDetail } from "./components/layout/NiftyDetail";
import { StockUniverse } from "./components/layout/StockUniverse";

const Screener = lazy(() =>
  import("./components/Screener").then((module) => ({ default: module.Screener })),
);
import type { StockSnapshot, Tick } from "./components/layout/types";
import type { MarketDataState } from "./market/quote";
import { useQuotesStore } from "./store/quotes";

type AlertTriggeredPayload = {
  alert_id: string;
  symbol: string;
  alert_type: string;
  value: number;
  threshold: number;
  timestamp: number;
};

type AlertBanner = AlertTriggeredPayload & {
  bannerId: string;
};

type BrokerAuthState = "CONNECTED" | "AUTHENTICATING" | "DISCONNECTED" | "ERROR";

type BrokerAuthPayload = {
  message?: string;
  state?: BrokerAuthState;
};

type IntegrityPayload = {
  trusted?: boolean;
};

export function App() {
  const [brokerAuth, setBrokerAuth] = useState<BrokerAuthState>("DISCONNECTED");
  const [brokerMessage, setBrokerMessage] = useState("Angel authentication not connected.");
  const [integrityWarning, setIntegrityWarning] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showScreener, setShowScreener] = useState(false);
  const [tick, setTick] = useState<Tick | null>(null);
  const [marketState, setMarketState] = useState<MarketDataState>("STARTING");
  const [selectedSymbol, setSelectedSymbol] = useState("NIFTY");
  const [selectedSnapshot, setSelectedSnapshot] = useState<StockSnapshot | null>(null);
  const [alertBanners, setAlertBanners] = useState<AlertBanner[]>([]);
  const alertBannerTimers = useRef<number[]>([]);
  const setQuote = useQuotesStore((state) => state.setQuote);

  useEffect(() => {
    let active = true;

    const loadRuntimeStatus = async () => {
      try {
        const [auth, integrity] = await Promise.all([
          invoke<BrokerAuthPayload>("get_broker_auth_status"),
          invoke<IntegrityPayload>("get_integrity_status"),
        ]);
        if (!active) {
          return;
        }
        setBrokerAuth(auth.state ?? "ERROR");
        setBrokerMessage(auth.message ?? "Angel authentication state unavailable.");
        setIntegrityWarning(integrity.trusted !== true);
      } catch {
        if (active) {
          setBrokerAuth("ERROR");
          setBrokerMessage("Angel authentication state unavailable.");
        }
      }
    };

    loadRuntimeStatus();
    const timer = window.setInterval(loadRuntimeStatus, 5_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    let active = true;

    const fetchQuote = async () => {
      try {
        const nextTick = await invoke<Tick>("get_quote", { symbol: "NIFTY" });
        if (!active) {
          return;
        }

        setTick(nextTick);
        setQuote(nextTick);
        setMarketState(nextTick.market_state);
      } catch {
        if (active) {
          setMarketState("DISCONNECTED");
        }
      }
    };

    fetchQuote();
    const timer = window.setInterval(fetchQuote, 1000);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [setQuote]);

  useEffect(() => {
    let active = true;

    const loadSnapshot = async () => {
      try {
        const snapshot = await invoke<StockSnapshot>("get_stock_snapshot", {
          symbol: selectedSymbol,
        });
        if (active) {
          setSelectedSnapshot(snapshot);
        }
      } catch {
        if (active) {
          setSelectedSnapshot(null);
        }
      }
    };

    loadSnapshot();
    const timer = window.setInterval(loadSnapshot, 300_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [selectedSymbol]);

  useEffect(() => {
    const unlisten = listen<AlertTriggeredPayload>("alert_triggered", (event) => {
      const banner: AlertBanner = {
        ...event.payload,
        bannerId: `${event.payload.alert_id}-${Date.now()}`,
      };

      setAlertBanners((current) => [banner, ...current].slice(0, 3));
      playAlertTone();

      const timeoutId = window.setTimeout(() => {
        setAlertBanners((current) =>
          current.filter((item) => item.bannerId !== banner.bannerId),
        );
      }, 5000);
      alertBannerTimers.current.push(timeoutId);
    });

    return () => {
      unlisten.then((dispose) => dispose()).catch(() => undefined);
      alertBannerTimers.current.forEach((timeoutId) => window.clearTimeout(timeoutId));
      alertBannerTimers.current = [];
    };
  }, []);

  const selectStock = (symbol: string, snapshot: StockSnapshot) => {
    setSelectedSymbol(symbol);
    setSelectedSnapshot(snapshot);
  };

  const selectNifty = () => {
    setSelectedSymbol("NIFTY");
  };

  return (
    <div style={shellStyle}>
      <ApexHeader
        brokerAuth={brokerAuth}
        integrityWarning={integrityWarning}
        marketState={marketState}
        onOpenScreener={() => setShowScreener((value) => !value)}
        onOpenSettings={() => setShowSettings((value) => !value)}
        onSelectNifty={selectNifty}
        selectedSymbol={selectedSymbol}
        tick={tick}
      />
      <main style={middleStyle}>
        <NiftyDetail selectedSymbol={selectedSymbol} snapshot={selectedSnapshot} />
        <StockUniverse
          onSelectSymbol={selectStock}
          selectedSymbol={selectedSymbol}
        />
      </main>
      <DetailTabs selectedSymbol={selectedSymbol} snapshot={selectedSnapshot} />
      <AlertBannerStack banners={alertBanners} />
      {showSettings ? (
        <SettingsPanel
          brokerAuth={brokerAuth}
          brokerMessage={brokerMessage}
          integrityWarning={integrityWarning}
        />
      ) : null}
      {showScreener ? (
        <div style={screenerOverlayStyle}>
          <div style={screenerPanelStyle}>
            <button
              aria-label="Close Screener"
              onClick={() => setShowScreener(false)}
              style={screenerCloseStyle}
              type="button"
            >
              Close
            </button>
            <Suspense fallback={<div style={settingsTitleStyle}>Loading screener...</div>}>
              <Screener />
            </Suspense>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function SettingsPanel({
  brokerAuth,
  brokerMessage,
  integrityWarning,
}: {
  brokerAuth: BrokerAuthState;
  brokerMessage: string;
  integrityWarning: boolean;
}) {
  return (
    <aside style={settingsStyle}>
      <div style={settingsTitleStyle}>Settings</div>
      <div style={settingsModeStyle}>Broker auth: {brokerAuth}</div>
      <div style={settingsMessageStyle}>{brokerMessage}</div>
      <div style={settingsModeStyle}>
        Release integrity: {integrityWarning ? "WARNING" : "TRUSTED"}
      </div>
      <div style={settingsHintStyle}>
        Credentials can only be provisioned through the local secure operator CLI.
      </div>
    </aside>
  );
}

function AlertBannerStack({ banners }: { banners: AlertBanner[] }) {
  if (banners.length === 0) {
    return null;
  }

  return (
    <div style={bannerStackStyle}>
      {banners.map((banner) => (
        <div key={banner.bannerId} style={bannerStyle}>
          <strong style={bannerTitleStyle}>Alert Triggered</strong>
          {banner.symbol} {banner.alert_type} alert: {formatAlertValue(banner.value)} crossed{" "}
          {formatAlertValue(banner.threshold)}
        </div>
      ))}
    </div>
  );
}

function formatAlertValue(value: number) {
  return new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: 2,
    minimumFractionDigits: 2,
  }).format(value);
}

function playAlertTone() {
  type AudioWindow = Window &
    typeof globalThis & {
      webkitAudioContext?: typeof AudioContext;
    };
  const AudioContextCtor = window.AudioContext ?? (window as AudioWindow).webkitAudioContext;

  if (!AudioContextCtor) {
    return;
  }

  try {
    const context = new AudioContextCtor();
    const oscillator = context.createOscillator();
    const gain = context.createGain();

    oscillator.frequency.value = 880;
    oscillator.type = "sine";
    gain.gain.value = 0.03;

    oscillator.connect(gain);
    gain.connect(context.destination);
    oscillator.start();
    oscillator.stop(context.currentTime + 0.12);

    window.setTimeout(() => {
      context.close().catch(() => undefined);
    }, 250);
  } catch {
    // Browser audio can be blocked until user gesture; banner still shows.
  }
}

const shellStyle = {
  background: "#0a0a0f",
  color: "#e8e8f0",
  display: "grid",
  fontFamily: "Inter, system-ui, sans-serif",
  gridTemplateRows: "80px minmax(0, 1fr) 300px",
  height: "100vh",
  minHeight: 0,
  overflow: "hidden",
  width: "100vw",
} satisfies React.CSSProperties;

const screenerOverlayStyle = {
  alignItems: "flex-start",
  background: "rgba(6,6,12,0.72)",
  display: "flex",
  inset: 0,
  justifyContent: "center",
  padding: "60px 24px 24px",
  position: "fixed",
  zIndex: 80,
} satisfies React.CSSProperties;

const screenerPanelStyle = {
  background: "#12121f",
  border: "1px solid #2a2a4a",
  borderRadius: 8,
  boxShadow: "0 18px 60px rgba(0,0,0,0.5)",
  maxHeight: "calc(100vh - 100px)",
  overflow: "auto",
  padding: 16,
  position: "relative",
  width: "min(960px, 100%)",
} satisfies React.CSSProperties;

const screenerCloseStyle = {
  background: "transparent",
  border: "1px solid #2a2a4a",
  borderRadius: 4,
  color: "#a0a0b8",
  cursor: "pointer",
  fontSize: 12,
  padding: "4px 10px",
  position: "absolute",
  right: 12,
  top: 12,
} satisfies React.CSSProperties;

const settingsStyle = {
  background: "#12121f",
  border: "1px solid #2a2a4a",
  boxShadow: "0 18px 60px rgba(0,0,0,0.42)",
  display: "grid",
  gap: 10,
  padding: 14,
  position: "fixed",
  right: 18,
  top: 88,
  width: 230,
  zIndex: 70,
} satisfies React.CSSProperties;

const settingsTitleStyle = {
  color: "#e8e8f0",
  fontSize: 14,
  fontWeight: 700,
} satisfies React.CSSProperties;

const settingsModeStyle = {
  color: "#9ca3af",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 11,
} satisfies React.CSSProperties;

const settingsMessageStyle = {
  color: "#ffd700",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 11,
  lineHeight: 1.45,
} satisfies React.CSSProperties;

const settingsHintStyle = {
  color: "#9ca3af",
  fontSize: 11,
  lineHeight: 1.45,
} satisfies React.CSSProperties;

const middleStyle = {
  display: "grid",
  gridTemplateColumns: "minmax(420px, 40%) minmax(560px, 60%)",
  minHeight: 0,
  minWidth: 0,
  overflow: "hidden",
} satisfies React.CSSProperties;

const bannerStackStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 10,
  maxWidth: 360,
  position: "fixed",
  right: 20,
  top: 94,
  width: "calc(100vw - 40px)",
  zIndex: 50,
} satisfies React.CSSProperties;

const bannerStyle = {
  background: "#12121f",
  border: "1px solid #ffd700",
  boxShadow: "0 14px 40px rgba(0, 0, 0, 0.35)",
  color: "#e8e8f0",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 12,
  lineHeight: 1.5,
  padding: "12px 14px",
} satisfies React.CSSProperties;

const bannerTitleStyle = {
  color: "#ffd700",
  display: "block",
  marginBottom: 4,
} satisfies React.CSSProperties;
