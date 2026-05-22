import axios, { AxiosError } from "axios";
import { toast } from "sonner";

import { useAuthStore } from "@/store/auth";

const baseURL = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

export const api = axios.create({
  baseURL,
  timeout: 30_000,
});

api.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (r) => r,
  (error: AxiosError<{ detail?: string }>) => {
    const status = error.response?.status;
    const detail = error.response?.data?.detail;

    if (status === 401) {
      useAuthStore.getState().logout();
      toast.error("نشست شما منقضی شده — لطفاً دوباره وارد شوید");
    } else if (status === 403) {
      toast.error(detail ?? "شما به این بخش دسترسی ندارید");
    } else if (status && status >= 500) {
      toast.error("خطای سرور — لطفاً بعداً تلاش کنید");
    } else if (detail) {
      toast.error(detail);
    }

    return Promise.reject(error);
  },
);

export type Role = "admin" | "manager" | "farm" | "livestock" | "kitchen";

export interface AuthUser {
  user_id: string;
  email: string;
  name: string;
  role: Role;
}

export interface TokenResponse extends AuthUser {
  access_token: string;
  token_type: string;
}

export interface DashboardKPIs {
  active_farms: number;
  active_cycles: number;
  livestock_groups: number;
  livestock_head: number;
  daily_milk_l: number;
  daily_eggs: number;
  inventory_items: number;
  low_stock_count: number;
  open_alerts: number;
  portions_today: number;
  avg_cost_per_portion: number | null;
}

export interface Alert {
  id: string;
  severity: "info" | "warning" | "error" | "success";
  section: string;
  title: string;
  message: string;
  ref_id: string | null;
  read: boolean;
  created_at: string;
}

export interface Farm {
  id: string;
  name: string;
  type: string;
  gps_lat: number | null;
  gps_lng: number | null;
  area_ha: number | null;
  notes: string | null;
}
