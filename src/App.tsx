import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

import { Alerts } from "./components/Alerts";
import { ChartPanel } from "./components/ChartPanel";
import { IVSurface } from "./components/IVSurface";
import { OptionsChain } from "./components/OptionsChain";
import { Screener } from "./components/Screener";
import { StrategyLab } from "./components/StrategyLab";
import { useQuotesStore } from "./store/quotes";

type Tick = {
  symbol: string;
  token: string;
  ltp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  oi: number;
  bid: number;
  ask: number;
  timestamp: number;
  exchange: string;
};

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

export function App() {
  const [tick, setTick] = useState<Tick | null>(null);
  const [isLive, setIsLive] = useState(false);
  const [alertBanners, setAlertBanners] = useState<AlertBanner[]>([]);
  const alertBannerTimers = useRef<number[]>([]);
  const setQuote = useQuotesStore((state) => state.setQuote);

  useEffect(() => {
    let active = true;

    const fetchQuote = async () => {
      try {
        const nextTick = await invoke<Tick>("get_quote", { symbol: "NIFTY" });
        if (!active) {
          return;
        }

        setTick(nextTick);
        setQuote({ ltp: nextTick.ltp, symbol: nextTick.symbol });
        setIsLive(true);
      } catch {
        if (active) {
          setIsLive(false);
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

  const priceText = useMemo(() => {
    if (!tick) {
      return "Connecting...";
    }

    return new Intl.NumberFormat("en-IN", {
      currency: "INR",
      maximumFractionDigits: 2,
      minimumFractionDigits: 2,
      style: "currency",
    }).format(tick.ltp);
  }, [tick]);

  return (
    <div
      style={{
        background: "#0a0a0a",
        color: "#ffffff",
        display: "flex",
        fontFamily: "monospace",
        minHeight: "100vh",
        overflow: "auto",
        width: "100vw",
      }}
    >
      <div
        style={{
          boxSizing: "border-box",
          display: "flex",
          flexDirection: "column",
          gap: 24,
          margin: "0 auto",
          maxWidth: 1180,
          padding: "32px 24px",
          width: "100%",
        }}
      >
        <header
          style={{
            alignItems: "baseline",
            display: "flex",
            flexWrap: "wrap",
            gap: "12px 18px",
            justifyContent: "space-between",
          }}
        >
          <div
            style={{
              color: "#ffffff",
              fontSize: 42,
              fontWeight: 700,
              letterSpacing: 0,
              lineHeight: 1.1,
              minHeight: 46,
            }}
          >
            {priceText}
          </div>
          <div
            style={{
              alignItems: "center",
              color: "#a3a3a3",
              display: "flex",
              fontSize: 13,
              gap: 8,
              letterSpacing: 0,
              minHeight: 20,
            }}
          >
            <span
              aria-label={isLive ? "live data" : "no data"}
              style={{
                background: isLive ? "#22c55e" : "#ef4444",
                borderRadius: "50%",
                display: "inline-block",
                height: 9,
                width: 9,
              }}
            />
            NIFTY 50
          </div>
        </header>
        <ChartPanel />
        <OptionsChain />
        <IVSurface />
        <Screener />
        <Alerts />
        <StrategyLab />
      </div>
      <AlertBannerStack banners={alertBanners} />
    </div>
  );
}

function AlertBannerStack({ banners }: { banners: AlertBanner[] }) {
  if (banners.length === 0) {
    return null;
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 10,
        maxWidth: 360,
        position: "fixed",
        right: 20,
        top: 20,
        width: "calc(100vw - 40px)",
        zIndex: 50,
      }}
    >
      {banners.map((banner) => (
        <div
          key={banner.bannerId}
          style={{
            background: "#1a1a2e",
            border: "1px solid #ffd700",
            borderRadius: 8,
            boxShadow: "0 14px 40px rgba(0, 0, 0, 0.35)",
            color: "#f8fafc",
            fontFamily: "monospace",
            fontSize: 12,
            lineHeight: 1.5,
            padding: "12px 14px",
          }}
        >
          <strong style={{ color: "#ffd700", display: "block", marginBottom: 4 }}>
            Alert Triggered
          </strong>
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
