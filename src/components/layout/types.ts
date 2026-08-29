export type StockSnapshot = {
  symbol: string;
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number;
  prev_close: number | null;
  change: number | null;
  change_pct: number | null;
  week_52_high: number | null;
  week_52_low: number | null;
  avg_volume_20d: number;
  volume_ratio: number | null;
  week_52_high_pct: number | null;
  week_52_low_pct: number | null;
  return_20d_pct?: number | null;
  relative_strength?: number | null;
  signal?: string;
  ltp?: number;
  matched_conditions?: string[];
  score?: number;
};

export type { Tick } from "../../market/quote";
