import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

export type MarketSession = {
  calendar_version: string;
  exchange: "NSE";
  session_date: string;
  is_trading_day: boolean;
  session_type: "NORMAL" | "HOLIDAY" | "SPECIAL" | "MUHURAT" | "HALF_SESSION";
  market_open: string | null;
  market_close: string | null;
  special_session: boolean;
  timing_status: string;
  current_state: "PRE_OPEN" | "OPEN" | "CLOSED" | "SCHEDULE_PENDING";
  next_session: {
    session_date: string;
    session_type: MarketSession["session_type"];
    market_open: string | null;
    market_close: string | null;
    special_session: boolean;
    timing_status: string;
  };
};

export function useMarketSession(): MarketSession | null {
  const [session, setSession] = useState<MarketSession | null>(null);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const value = await invoke<MarketSession>("get_market_session");
        if (active) setSession(value);
      } catch {
        if (active) setSession(null);
      }
    };
    refresh();
    const timer = window.setInterval(refresh, 30_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  return session;
}
