import { create } from "zustand";

export type ChainLeg = {
  symbol: string;
  expiry: string;
  strike: number;
};

export type ChainState = {
  chains: Record<string, ChainLeg[]>;
};

export const useChainStore = create<ChainState>(() => ({
  chains: {},
}));
