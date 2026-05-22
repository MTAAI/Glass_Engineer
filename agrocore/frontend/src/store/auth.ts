import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { AuthUser, Role } from "@/lib/api";

interface AuthState {
  token: string | null;
  user: AuthUser | null;
  setSession: (token: string, user: AuthUser) => void;
  logout: () => void;
  hasRole: (...roles: Role[]) => boolean;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      user: null,
      setSession: (token, user) => set({ token, user }),
      logout: () => set({ token: null, user: null }),
      hasRole: (...roles) => {
        const u = get().user;
        if (!u) return false;
        if (u.role === "admin") return true;
        return roles.includes(u.role);
      },
    }),
    { name: "agrocore_auth" },
  ),
);
