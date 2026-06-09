import { create } from "zustand";

export type Quote = {
  symbol: string;
  ltp: number;
};

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
