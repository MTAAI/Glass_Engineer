import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Map,
  Sprout,
  Beef,
  Fish,
  Factory,
  ChefHat,
  CalendarDays,
  ShieldCheck,
  Tractor,
  HardHat,
  Package,
  Coins,
  BarChart3,
  Camera,
  Settings,
  LogOut,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { useAuthStore } from "@/store/auth";

interface MenuItem {
  to: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  group?: string;
  color?: string;
}

const MENU: MenuItem[] = [
  { to: "/", label: "داشبورد", icon: LayoutDashboard },
  { to: "/map", label: "نقشه", icon: Map },

  { to: "/farm", label: "کشت و زراعت", icon: Sprout, group: "تولید", color: "text-primary-600" },
  { to: "/livestock", label: "دامداری", icon: Beef, group: "تولید", color: "text-livestock" },
  { to: "/aquaculture", label: "آبزی‌پروری", icon: Fish, group: "تولید", color: "text-water" },
  { to: "/processing", label: "فراوری", icon: Factory, group: "تولید", color: "text-processing" },

  { to: "/kitchen", label: "آشپزخانه", icon: ChefHat, group: "آشپزخانه", color: "text-kitchen" },
  { to: "/menu", label: "منوی ۳۶۵", icon: CalendarDays, group: "آشپزخانه" },
  { to: "/haccp", label: "HACCP", icon: ShieldCheck, group: "آشپزخانه" },

  { to: "/machinery", label: "ماشین‌آلات", icon: Tractor, group: "عملیاتی" },
  { to: "/workers", label: "کارگران", icon: HardHat, group: "عملیاتی" },
  { to: "/inventory", label: "انبار", icon: Package, group: "عملیاتی" },

  { to: "/cost", label: "قیمت تمام‌شده", icon: Coins, group: "مالی", color: "text-soil" },
  { to: "/reports", label: "گزارش‌ها", icon: BarChart3, group: "مالی" },

  { to: "/camera", label: "دوربین", icon: Camera, group: "سیستم" },
  { to: "/settings", label: "تنظیمات", icon: Settings, group: "سیستم" },
];

function groupMenu(items: MenuItem[]) {
  const ungrouped: MenuItem[] = [];
  const groups: Record<string, MenuItem[]> = {};
  for (const item of items) {
    if (!item.group) ungrouped.push(item);
    else (groups[item.group] ||= []).push(item);
  }
  return { ungrouped, groups };
}

export function Sidebar() {
  const { ungrouped, groups } = groupMenu(MENU);
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);

  return (
    <aside className="hidden lg:flex w-64 shrink-0 flex-col bg-white border-l border-slate-200 h-screen sticky top-0">
      <div className="px-5 py-4 border-b border-slate-200">
        <div className="flex items-center gap-2">
          <div className="w-9 h-9 rounded-lg bg-primary-600 text-white grid place-items-center font-bold">🌿</div>
          <div>
            <div className="font-bold text-slate-900">AgroCore OS</div>
            <div className="text-xs text-slate-500">سامانه کشت‌وصنعت</div>
          </div>
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-3 space-y-1">
        {ungrouped.map((item) => (
          <NavItem key={item.to} item={item} />
        ))}

        {Object.entries(groups).map(([groupName, items]) => (
          <div key={groupName} className="pt-3">
            <div className="px-3 pb-1 text-[11px] font-semibold uppercase text-slate-400 tracking-wider">
              — {groupName} —
            </div>
            {items.map((item) => (
              <NavItem key={item.to} item={item} />
            ))}
          </div>
        ))}
      </nav>

      {user && (
        <div className="border-t border-slate-200 px-3 py-3">
          <div className="flex items-center gap-2 px-2 py-2">
            <div className="w-9 h-9 rounded-full bg-slate-200 grid place-items-center text-slate-700 font-bold">
              {user.name.charAt(0)}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium truncate">{user.name}</div>
              <div className="text-xs text-slate-500">{roleLabel(user.role)}</div>
            </div>
            <button
              type="button"
              onClick={logout}
              className="p-2 rounded-lg hover:bg-slate-100 text-slate-500"
              title="خروج"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </aside>
  );
}

function NavItem({ item }: { item: MenuItem }) {
  const Icon = item.icon;
  return (
    <NavLink
      to={item.to}
      end={item.to === "/"}
      className={({ isActive }) =>
        cn(
          "flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition",
          isActive
            ? "bg-primary-50 text-primary-700 font-medium"
            : "text-slate-700 hover:bg-slate-100",
        )
      }
    >
      <Icon className={cn("w-4 h-4", item.color)} />
      <span>{item.label}</span>
    </NavLink>
  );
}

function roleLabel(role: string): string {
  return (
    {
      admin: "مدیر کل",
      manager: "مدیر اجرایی",
      farm: "مدیر مزرعه",
      livestock: "مدیر دامداری",
      kitchen: "سرآشپز",
    } as Record<string, string>
  )[role] ?? role;
}
