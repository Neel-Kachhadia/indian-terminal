import { create } from "zustand";

export type AlertType = "price" | "oi" | "pcr";
export type AlertCondition = "above" | "below" | "crosses_up" | "crosses_down";

export type Alert = {
  id: string;
  symbol: string;
  alert_type: AlertType;
  threshold: number;
  condition: AlertCondition;
  active: boolean;
  triggered_at: number | null;
  created_at: number;
};

export type AlertsState = {
  alerts: Alert[];
  removeAlert: (id: string) => void;
  setAlerts: (alerts: Alert[]) => void;
};

export const useAlertsStore = create<AlertsState>((set) => ({
  alerts: [],
  removeAlert: (id) =>
    set((state) => ({
      alerts: state.alerts.filter((alert) => alert.id !== id),
    })),
  setAlerts: (alerts) => set({ alerts }),
}));
