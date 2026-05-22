import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import { Plus, X, Milk, Egg, Beef, ChevronRight } from "lucide-react";

import { api } from "@/lib/api";
import { fa, tomans } from "@/lib/utils";
import { PageHeader } from "@/components/PageHeader";

interface Group {
  id: string;
  farm_id: string;
  species: string;
  purpose: string;
  name: string;
  current_count: number;
  target_count: number;
  entry_date: string | null;
  total_cost_to_date: number | null;
  notes: string | null;
}

interface Input {
  id: string;
  input_category: string;
  input_name: string;
  date: string;
  quantity: number;
  unit: string;
  unit_price: number;
  total_price: number;
}

interface Production {
  id: string;
  date: string;
  production_type: string;
  amount: number;
}

interface CostResult {
  group_id: string;
  name: string;
  species: string;
  breakdown: {
    direct_materials: number;
    direct_labor: number;
    machinery: number;
    utilities: number;
    overhead: number;
    total: number;
    by_subcategory: Record<string, number>;
  };
  production: Record<string, number>;
  cost_per_unit: Record<string, number>;
}

const SPECIES_LABEL: Record<string, { label: string; icon: string }> = {
  cow: { label: "گاو", icon: "🐄" },
  sheep: { label: "گوسفند", icon: "🐑" },
  goat: { label: "بز", icon: "🐐" },
  chicken: { label: "مرغ", icon: "🐔" },
  duck: { label: "اردک", icon: "🦆" },
  goose: { label: "غاز", icon: "🦢" },
  turkey: { label: "بوقلمون", icon: "🦃" },
  trout: { label: "قزل‌آلا", icon: "🐟" },
  carp: { label: "کپور", icon: "🐠" },
  tilapia: { label: "تیلاپیا", icon: "🐟" },
};

const PURPOSE_LABEL: Record<string, string> = {
  dairy: "شیری",
  meat: "گوشتی",
  egg: "تخم‌گذار",
  dual: "دو منظوره",
  wool: "پشمی",
};

const PRODUCTION_LABEL: Record<string, { label: string; unit: string }> = {
  milk: { label: "شیر", unit: "لیتر" },
  egg: { label: "تخم", unit: "عدد" },
  weight_gain: { label: "افزایش وزن", unit: "کیلو" },
  wool: { label: "پشم", unit: "کیلو" },
};

export function LivestockPage() {
  const [selected, setSelected] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);

  const { data: groups = [] } = useQuery({
    queryKey: ["livestock-groups"],
    queryFn: async () => (await api.get<Group[]>("/api/v1/livestock/groups")).data,
  });

  const selectedGroup = groups.find((g) => g.id === selected) ?? groups[0];

  return (
    <div>
      <PageHeader
        title="دامداری و آبزی‌پروری"
        subtitle="گاو، گوسفند، مرغ، اردک، غاز، بوقلمون، ماهی"
        actions={
          <button onClick={() => setShowCreate(!showCreate)} className="btn-primary">
            {showCreate ? <X className="w-4 h-4" /> : <Plus className="w-4 h-4" />}
            {showCreate ? "بستن" : "گروه جدید"}
          </button>
        }
      />

      {showCreate && <CreateGroupForm onClose={() => setShowCreate(false)} />}

      <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-4">
        <div className="card p-2">
          <div className="space-y-1">
            {groups.length === 0 && (
              <div className="text-xs text-slate-400 p-6 text-center">گروه دامی ثبت نشده</div>
            )}
            {groups.map((g) => {
              const meta = SPECIES_LABEL[g.species] ?? { label: g.species, icon: "🐾" };
              return (
                <button
                  key={g.id}
                  onClick={() => setSelected(g.id)}
                  className={`w-full text-right px-3 py-2.5 rounded-lg text-sm transition ${
                    selectedGroup?.id === g.id ? "bg-livestock/10 text-livestock font-medium" : "hover:bg-slate-100"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-lg">{meta.icon}</span>
                    <div className="flex-1 min-w-0">
                      <div className="truncate">{g.name}</div>
                      <div className="text-[11px] text-slate-500 mt-0.5">
                        {PURPOSE_LABEL[g.purpose] ?? g.purpose} · <span className="num">{fa(g.current_count)}</span> {g.species === "chicken" || g.species === "trout" ? "قطعه" : "رأس"}
                      </div>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {selectedGroup && (
          <div className="space-y-4">
            <GroupOverview group={selectedGroup} />
            <DailyProduction groupId={selectedGroup.id} species={selectedGroup.species} purpose={selectedGroup.purpose} />
            <InputsLedger groupId={selectedGroup.id} />
            <CostPanel groupId={selectedGroup.id} />
          </div>
        )}
      </div>
    </div>
  );
}

function CreateGroupForm({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const { register, handleSubmit, reset } = useForm({
    defaultValues: {
      farm_id: "",
      species: "cow",
      purpose: "dairy",
      name: "",
      current_count: 0,
      target_count: 0,
      entry_date: "",
      notes: "",
    },
  });

  const { data: farms = [] } = useQuery({
    queryKey: ["farms"],
    queryFn: async () => (await api.get("/api/v1/farms")).data,
  });

  const mutate = useMutation({
    mutationFn: async (d: any) => (await api.post("/api/v1/livestock/groups", { ...d, entry_date: d.entry_date || null })).data,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["livestock-groups"] });
      toast.success("گروه دامی ایجاد شد");
      reset();
      onClose();
    },
  });

  return (
    <form
      onSubmit={handleSubmit((d) => mutate.mutate(d))}
      className="card p-4 mb-4 grid grid-cols-1 sm:grid-cols-3 gap-3"
    >
      <div>
        <label className="label">مزرعه/کارخانه</label>
        <select className="input" {...register("farm_id", { required: true })}>
          <option value="">— انتخاب —</option>
          {farms.map((f: any) => <option key={f.id} value={f.id}>{f.name}</option>)}
        </select>
      </div>
      <div>
        <label className="label">گونه</label>
        <select className="input" {...register("species")}>
          {Object.entries(SPECIES_LABEL).map(([k, v]) => (
            <option key={k} value={k}>{v.icon} {v.label}</option>
          ))}
        </select>
      </div>
      <div>
        <label className="label">هدف</label>
        <select className="input" {...register("purpose")}>
          {Object.entries(PURPOSE_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
      </div>
      <div className="sm:col-span-2">
        <label className="label">نام گروه</label>
        <input className="input" placeholder="مثلاً: گله گاو شیری ۱" {...register("name", { required: true })} />
      </div>
      <div>
        <label className="label">تاریخ ورود</label>
        <input type="date" className="input" {...register("entry_date")} />
      </div>
      <div>
        <label className="label">جمعیت فعلی</label>
        <input type="number" className="input" {...register("current_count", { valueAsNumber: true })} />
      </div>
      <div>
        <label className="label">جمعیت هدف</label>
        <input type="number" className="input" {...register("target_count", { valueAsNumber: true })} />
      </div>
      <div className="sm:col-span-3 flex justify-end gap-2">
        <button type="button" onClick={onClose} className="btn-secondary">انصراف</button>
        <button type="submit" className="btn-primary" disabled={mutate.isPending}>ساخت گروه</button>
      </div>
    </form>
  );
}

function GroupOverview({ group }: { group: Group }) {
  const meta = SPECIES_LABEL[group.species] ?? { label: group.species, icon: "🐾" };
  return (
    <div className="card p-4">
      <div className="flex items-center gap-3 mb-3">
        <div className="text-4xl">{meta.icon}</div>
        <div className="flex-1">
          <h3 className="font-bold text-lg">{group.name}</h3>
          <div className="text-sm text-slate-500">{meta.label} {PURPOSE_LABEL[group.purpose] ?? group.purpose}</div>
        </div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Stat label="جمعیت فعلی" value={fa(group.current_count)} />
        <Stat label="جمعیت هدف" value={fa(group.target_count)} />
        <Stat label="تاریخ ورود" value={group.entry_date ?? "—"} />
        <Stat label="هزینه تجمعی" value={group.total_cost_to_date ? tomans(group.total_cost_to_date) : "—"} />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-50 rounded-lg p-3">
      <div className="text-[11px] text-slate-500">{label}</div>
      <div className="text-sm font-bold mt-0.5 num">{value}</div>
    </div>
  );
}

function DailyProduction({ groupId, species, purpose }: { groupId: string; species: string; purpose: string }) {
  const qc = useQueryClient();
  const { data: rows = [] } = useQuery({
    queryKey: ["livestock-production", groupId],
    queryFn: async () => (await api.get<Production[]>(`/api/v1/livestock/groups/${groupId}/production`)).data,
  });

  const defaultType = purpose === "dairy" ? "milk" : purpose === "egg" ? "egg" : "weight_gain";
  const { register, handleSubmit, reset } = useForm({
    defaultValues: { date: new Date().toISOString().slice(0, 10), production_type: defaultType, amount: 0 },
  });

  const mutate = useMutation({
    mutationFn: async (d: any) => (await api.post(`/api/v1/livestock/groups/${groupId}/production`, d)).data,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["livestock-production", groupId] });
      qc.invalidateQueries({ queryKey: ["livestock-cost", groupId] });
      toast.success("تولید روزانه ثبت شد");
      reset({ date: new Date().toISOString().slice(0, 10), production_type: defaultType, amount: 0 });
    },
  });

  const Icon = purpose === "dairy" ? Milk : purpose === "egg" ? Egg : Beef;

  return (
    <div className="card p-4">
      <h3 className="font-semibold flex items-center gap-2 mb-3">
        <Icon className="w-4 h-4 text-livestock" /> تولید روزانه
      </h3>
      <form
        onSubmit={handleSubmit((d) => mutate.mutate(d))}
        className="grid grid-cols-2 sm:grid-cols-4 gap-2 p-3 bg-slate-50 rounded-lg mb-3"
      >
        <input type="date" className="input" {...register("date", { required: true })} />
        <select className="input" {...register("production_type")}>
          <option value="milk">شیر (لیتر)</option>
          <option value="egg">تخم (عدد)</option>
          <option value="weight_gain">وزن (کیلو)</option>
          <option value="wool">پشم (کیلو)</option>
        </select>
        <input type="number" step="0.01" placeholder="مقدار" className="input" {...register("amount", { valueAsNumber: true, required: true })} />
        <button type="submit" className="btn-primary" disabled={mutate.isPending}>ثبت</button>
      </form>

      {rows.length === 0 ? (
        <div className="text-sm text-slate-400 text-center py-4">رکورد تولیدی ثبت نشده</div>
      ) : (
        <div className="overflow-x-auto max-h-72 overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="text-xs text-slate-500 border-b sticky top-0 bg-white">
              <tr>
                <th className="text-right py-2 px-2">تاریخ</th>
                <th className="text-right py-2 px-2">نوع</th>
                <th className="text-right py-2 px-2">مقدار</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {rows.map((r) => {
                const t = PRODUCTION_LABEL[r.production_type] ?? { label: r.production_type, unit: "" };
                return (
                  <tr key={r.id}>
                    <td className="py-1.5 px-2 num">{r.date}</td>
                    <td className="py-1.5 px-2">{t.label}</td>
                    <td className="py-1.5 px-2 num font-medium">{fa(r.amount)} {t.unit}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function InputsLedger({ groupId }: { groupId: string }) {
  const qc = useQueryClient();
  const { data: rows = [] } = useQuery({
    queryKey: ["livestock-inputs", groupId],
    queryFn: async () => (await api.get<Input[]>(`/api/v1/livestock/groups/${groupId}/inputs`)).data,
  });

  const { register, handleSubmit, reset, watch } = useForm({
    defaultValues: {
      input_category: "feed",
      input_name: "",
      date: new Date().toISOString().slice(0, 10),
      quantity: 0,
      unit: "kg",
      unit_price: 0,
    },
  });

  const total = (watch("quantity") || 0) * (watch("unit_price") || 0);

  const mutate = useMutation({
    mutationFn: async (d: any) => (await api.post(`/api/v1/livestock/groups/${groupId}/inputs`, d)).data,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["livestock-inputs", groupId] });
      qc.invalidateQueries({ queryKey: ["livestock-cost", groupId] });
      qc.invalidateQueries({ queryKey: ["livestock-groups"] });
      toast.success("هزینه ثبت شد");
      reset({ input_category: "feed", input_name: "", date: new Date().toISOString().slice(0, 10), quantity: 0, unit: "kg", unit_price: 0 });
    },
  });

  return (
    <div className="card p-4">
      <h3 className="font-semibold mb-3">هزینه‌ها و نهاده‌ها</h3>

      <form
        onSubmit={handleSubmit((d) => mutate.mutate(d))}
        className="grid grid-cols-2 lg:grid-cols-7 gap-2 p-3 bg-slate-50 rounded-lg mb-3"
      >
        <select className="input" {...register("input_category")}>
          <option value="feed">خوراک</option>
          <option value="medicine">دارو</option>
          <option value="vet">دامپزشک</option>
          <option value="bedding">بستر</option>
          <option value="utilities">آب/برق</option>
          <option value="overhead">سربار</option>
        </select>
        <input className="input lg:col-span-2" placeholder="کنسانتره ۱۸٪" {...register("input_name", { required: true })} />
        <input type="date" className="input" {...register("date")} />
        <input type="number" step="0.001" className="input" placeholder="مقدار" {...register("quantity", { valueAsNumber: true })} />
        <input type="number" className="input" placeholder="قیمت واحد" {...register("unit_price", { valueAsNumber: true })} />
        <button type="submit" className="btn-primary" disabled={mutate.isPending}>
          <span className="text-xs">جمع: {fa(total)}</span>
        </button>
      </form>

      {rows.length === 0 ? (
        <div className="text-sm text-slate-400 text-center py-4">هزینه‌ای ثبت نشده</div>
      ) : (
        <div className="overflow-x-auto max-h-64 overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="text-xs text-slate-500 border-b sticky top-0 bg-white">
              <tr>
                <th className="text-right py-2 px-2">تاریخ</th>
                <th className="text-right py-2 px-2">دسته</th>
                <th className="text-right py-2 px-2">نام</th>
                <th className="text-right py-2 px-2">مقدار</th>
                <th className="text-right py-2 px-2">جمع</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {rows.map((r) => (
                <tr key={r.id}>
                  <td className="py-1.5 px-2 num text-slate-600">{r.date}</td>
                  <td className="py-1.5 px-2"><span className="text-xs bg-slate-100 px-2 py-0.5 rounded">{r.input_category}</span></td>
                  <td className="py-1.5 px-2">{r.input_name}</td>
                  <td className="py-1.5 px-2 num">{fa(r.quantity)} {r.unit}</td>
                  <td className="py-1.5 px-2 num font-medium">{tomans(r.total_price)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function CostPanel({ groupId }: { groupId: string }) {
  const { data } = useQuery({
    queryKey: ["livestock-cost", groupId],
    queryFn: async () => (await api.get<CostResult>(`/api/v1/livestock/groups/${groupId}/cost`)).data,
  });

  if (!data) return null;

  const b = data.breakdown;
  const total = b.total || 1;

  const buckets = [
    { label: "خوراک و دارو", color: "bg-livestock", amount: b.direct_materials },
    { label: "کارگر", color: "bg-amber-500", amount: b.direct_labor },
    { label: "آب و برق", color: "bg-water", amount: b.utilities },
    { label: "سربار", color: "bg-slate-500", amount: b.overhead },
  ].filter((x) => x.amount > 0);

  return (
    <div className="card p-4">
      <h3 className="font-semibold mb-3">قیمت تمام‌شده تجمعی</h3>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
        <div className="card p-3 border-t-4 border-soil">
          <div className="text-xs text-slate-500">جمع هزینه</div>
          <div className="text-xl font-bold mt-1 num">{tomans(b.total)}</div>
        </div>
        <div className="card p-3 border-t-4 border-livestock">
          <div className="text-xs text-slate-500">قیمت هر واحد تولید</div>
          {Object.keys(data.cost_per_unit).length === 0 ? (
            <div className="text-sm text-slate-400 mt-1">هنوز تولیدی ثبت نشده</div>
          ) : (
            <div className="text-sm space-y-0.5 mt-1">
              {Object.entries(data.cost_per_unit).map(([k, v]) => {
                const meta = PRODUCTION_LABEL[k] ?? { label: k, unit: "" };
                return (
                  <div key={k} className="flex justify-between">
                    <span className="text-slate-600">هر {meta.unit} {meta.label}:</span>
                    <span className="num font-bold">{tomans(v)}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      <div className="space-y-2">
        {buckets.length === 0 ? (
          <div className="text-sm text-slate-400 text-center py-2">هنوز هزینه‌ای ثبت نشده</div>
        ) : (
          buckets.map((x) => {
            const pct = (x.amount / total) * 100;
            return (
              <div key={x.label}>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="font-medium text-slate-700">{x.label}</span>
                  <span className="num text-slate-600">{tomans(x.amount)} <span className="text-slate-400">({fa(Math.round(pct))}٪)</span></span>
                </div>
                <div className="h-2 bg-slate-100 rounded overflow-hidden">
                  <div className={`h-full ${x.color}`} style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
