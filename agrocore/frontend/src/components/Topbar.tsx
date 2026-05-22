import { Bell, Camera, Search, MapPin } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { api, type Alert } from "@/lib/api";
import { cn } from "@/lib/utils";

export function Topbar() {
  const [open, setOpen] = useState(false);

  const { data: alerts = [] } = useQuery({
    queryKey: ["alerts", "unread"],
    queryFn: async () => {
      const r = await api.get<Alert[]>("/api/v1/alerts", { params: { unread_only: true } });
      return r.data;
    },
    refetchInterval: 60_000,
  });

  return (
    <header className="sticky top-0 z-30 bg-white border-b border-slate-200">
      <div className="flex items-center gap-3 px-4 py-2.5">
        <div className="flex-1 max-w-md relative">
          <Search className="w-4 h-4 absolute right-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="جستجو در سامانه..."
            className="input pr-9"
          />
        </div>

        <button type="button" className="btn-ghost" title="GPS موقعیت">
          <MapPin className="w-4 h-4" />
        </button>

        <button type="button" className="btn-ghost" title="دوربین">
          <Camera className="w-4 h-4" />
        </button>

        <div className="relative">
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="btn-ghost relative"
            title="هشدارها"
          >
            <Bell className="w-4 h-4" />
            {alerts.length > 0 && (
              <span className="absolute -top-1 -right-1 w-5 h-5 grid place-items-center text-[10px] font-bold rounded-full bg-red-500 text-white">
                {alerts.length > 9 ? "+9" : alerts.length}
              </span>
            )}
          </button>

          {open && (
            <div className="absolute left-0 mt-2 w-80 card overflow-hidden z-40">
              <div className="px-3 py-2 border-b border-slate-200 font-medium text-sm">
                هشدارهای جدید ({alerts.length})
              </div>
              <div className="max-h-80 overflow-y-auto divide-y divide-slate-100">
                {alerts.length === 0 ? (
                  <div className="px-4 py-6 text-center text-sm text-slate-500">هشدار جدیدی وجود ندارد</div>
                ) : (
                  alerts.map((a) => (
                    <div key={a.id} className="px-3 py-2.5 hover:bg-slate-50">
                      <div className="flex items-start gap-2">
                        <span
                          className={cn(
                            "w-2 h-2 rounded-full mt-1.5 shrink-0",
                            a.severity === "error" && "bg-red-500",
                            a.severity === "warning" && "bg-amber-500",
                            a.severity === "info" && "bg-sky-500",
                            a.severity === "success" && "bg-emerald-500",
                          )}
                        />
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium">{a.title}</div>
                          <div className="text-xs text-slate-600 mt-0.5">{a.message}</div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
