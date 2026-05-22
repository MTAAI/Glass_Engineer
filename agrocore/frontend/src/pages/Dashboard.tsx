import { useQuery } from "@tanstack/react-query";
import {
  Sprout,
  Beef,
  ChefHat,
  Package,
  Bell,
  Egg,
  Milk,
  Coins,
} from "lucide-react";

import { api, type DashboardKPIs } from "@/lib/api";
import { fa, tomans } from "@/lib/utils";
import { PageHeader } from "@/components/PageHeader";

export function DashboardPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["dashboard", "kpis"],
    queryFn: async () => (await api.get<DashboardKPIs>("/api/v1/dashboard/kpis")).data,
    refetchInterval: 60_000,
  });

  return (
    <div>
      <PageHeader
        title="داشبورد مدیریتی"
        subtitle="نگاهی به وضعیت کل کشت‌وصنعت در یک نگاه"
      />

      {isLoading || !data ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="kpi-card border-slate-200">
              <div className="h-16 bg-slate-100 rounded animate-pulse" />
            </div>
          ))}
        </div>
      ) : (
        <>
          <section className="mb-6">
            <h2 className="text-sm font-semibold text-slate-500 mb-3">مزرعه</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <KpiCard icon={Sprout} accent="border-primary-600" label="مزارع فعال" value={fa(data.active_farms)} hint="مزرعه و کارخانه" />
              <KpiCard icon={Sprout} accent="border-primary-600" label="چرخه‌های کشت فعال" value={fa(data.active_cycles)} hint="در حال داشت" />
              <KpiCard icon={Package} accent="border-primary-600" label="موجودی انبار" value={fa(data.inventory_items)} hint="کالا ثبت‌شده" />
              <KpiCard
                icon={Bell}
                accent={data.low_stock_count > 0 ? "border-red-500" : "border-primary-600"}
                label="هشدار کمبود"
                value={fa(data.low_stock_count)}
                hint="موجودی زیر حداقل"
              />
            </div>
          </section>

          <section className="mb-6">
            <h2 className="text-sm font-semibold text-slate-500 mb-3">دام و طیور</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <KpiCard icon={Beef} accent="border-livestock" label="گروه‌های دامی" value={fa(data.livestock_groups)} />
              <KpiCard icon={Beef} accent="border-livestock" label="جمعیت دام" value={fa(data.livestock_head)} hint="رأس / قطعه" />
              <KpiCard icon={Milk} accent="border-livestock" label="شیر امروز" value={`${fa(data.daily_milk_l)} لیتر`} />
              <KpiCard icon={Egg} accent="border-livestock" label="تخم‌مرغ امروز" value={fa(data.daily_eggs)} />
            </div>
          </section>

          <section className="mb-6">
            <h2 className="text-sm font-semibold text-slate-500 mb-3">آشپزخانه و مالی</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <KpiCard icon={ChefHat} accent="border-kitchen" label="پرس امروز" value={fa(data.portions_today)} />
              <KpiCard
                icon={Coins}
                accent="border-soil"
                label="میانگین هزینه پرس"
                value={data.avg_cost_per_portion ? tomans(data.avg_cost_per_portion) : "—"}
              />
              <KpiCard
                icon={Bell}
                accent={data.open_alerts > 0 ? "border-amber-500" : "border-slate-300"}
                label="هشدارهای باز"
                value={fa(data.open_alerts)}
              />
              <div className="kpi-card border-primary-600">
                <div className="text-xs text-slate-500">وضعیت سیستم</div>
                <div className="text-lg font-bold text-primary-700 mt-1">✓ فعال</div>
                <div className="text-xs text-slate-400 mt-1">نسخه ۱.۰ — فاز ۰</div>
              </div>
            </div>
          </section>

          <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="card p-5">
              <h3 className="font-semibold mb-2">نقشه راه توسعه</h3>
              <p className="text-sm text-slate-500 mb-3">پیشرفت پیاده‌سازی AgroCore OS</p>
              <ul className="space-y-2 text-sm">
                <Phase done label="فاز ۰: پایه‌گذاری (Auth, DB, Layout)" />
                <Phase done label="فاز ۱: هسته مزرعه + موتور هزینه" />
                <Phase done label="فاز ۲: دامداری + آبزی‌پروری" />
                <Phase label="فاز ۳: فراوری — API آماده" />
                <Phase done label="فاز ۴: انبار یکپارچه" />
                <Phase label="فاز ۵: آشپزخانه" />
                <Phase label="فاز ۶: مالی و گزارش" />
                <Phase label="فاز ۷: ابزارها و تکمیل" />
              </ul>
            </div>

            <div className="card p-5">
              <h3 className="font-semibold mb-2">قوانین کسب‌وکار</h3>
              <p className="text-sm text-slate-500 mb-3">محدودیت‌های اعمال‌شده سامانه</p>
              <ul className="space-y-2 text-sm text-slate-700">
                <li className="flex items-start gap-2"><span className="text-water">🐟</span> قزل‌آلا در تیر/مرداد/شهریور ممنوع</li>
                <li className="flex items-start gap-2"><span className="text-livestock">🐑</span> حداکثر گوسفند: ۲۰۰ رأس</li>
                <li className="flex items-start gap-2"><span className="text-kitchen">🐟</span> حداقل ۴ وعده ماهی ماهانه الزامی</li>
                <li className="flex items-start gap-2"><span className="text-kitchen">🍗</span> دمای پخت مرغ: حداقل ۷۵ درجه</li>
                <li className="flex items-start gap-2"><span className="text-soil">💰</span> قیمت مواد خودی = قیمت تمام‌شده تولید</li>
              </ul>
            </div>
          </section>
        </>
      )}
    </div>
  );
}

function KpiCard({
  icon: Icon,
  accent,
  label,
  value,
  hint,
}: {
  icon: React.ComponentType<{ className?: string }>;
  accent: string;
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className={`kpi-card ${accent}`}>
      <div className="flex items-start justify-between">
        <div className="min-w-0">
          <div className="text-xs text-slate-500">{label}</div>
          <div className="text-2xl font-bold mt-1 truncate">{value}</div>
          {hint && <div className="text-[11px] text-slate-400 mt-0.5">{hint}</div>}
        </div>
        <Icon className="w-5 h-5 text-slate-400 shrink-0" />
      </div>
    </div>
  );
}

function Phase({ label, done }: { label: string; done?: boolean }) {
  return (
    <li className="flex items-center gap-2">
      <span
        className={`w-4 h-4 rounded grid place-items-center text-[10px] ${
          done ? "bg-primary-600 text-white" : "bg-slate-200 text-slate-400"
        }`}
      >
        {done ? "✓" : ""}
      </span>
      <span className={done ? "text-slate-700" : "text-slate-500"}>{label}</span>
    </li>
  );
}
