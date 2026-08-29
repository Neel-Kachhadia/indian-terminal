import { create } from "zustand";
import type { Tick } from "../market/quote";

export type Quote = Tick;

export type QuotesState = {
  quotes: Record<string, Quote>;
  setQuote: (quote: Quote) => void;
};

export const useQuotesStore = create<QuotesState>((set) => ({
  quotes: {},
  setQuote: (quote) =>
    set((state) => ({
      quotes: {
        ...state.quotes,
        [quote.symbol]: quote,
      },
    })),
}));
