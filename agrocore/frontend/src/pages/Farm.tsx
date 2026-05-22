import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import { Plus, X, MapPin, Sprout, Calendar, Tractor, ChevronRight } from "lucide-react";

import { api, type Farm } from "@/lib/api";
import { fa, tomans } from "@/lib/utils";
import { PageHeader } from "@/components/PageHeader";

interface Block {
  id: string;
  farm_id: string;
  name: string;
  area_ha: number;
  soil_type: string | null;
  irrigation_type: string | null;
  notes: string | null;
  active: boolean;
}

interface CropType {
  id: string;
  name: string;
  name_fa: string;
  category: string | null;
}

interface Cycle {
  id: string;
  block_id: string;
  block_name: string | null;
  crop_type_id: string;
  crop_name: string | null;
  season: string;
  year: number;
  plant_date: string | null;
  expected_harvest_date: string | null;
  actual_harvest_date: string | null;
  target_yield_kg: number | null;
  actual_yield_kg: number | null;
  status: string;
  total_cost_tomans: number | null;
  cost_per_kg: number | null;
  notes: string | null;
}

const STATUS_LABEL: Record<string, string> = {
  planned: "برنامه‌ریزی‌شده",
  planting: "در حال کاشت",
  growing: "در حال داشت",
  harvested: "برداشت‌شده",
  closed: "بسته‌شده",
};

const SEASON_LABEL: Record<string, string> = {
  spring: "بهار",
  summer: "تابستان",
  fall: "پاییز",
  winter: "زمستان",
};

export function FarmPage() {
  const [selectedFarm, setSelectedFarm] = useState<string | null>(null);
  const [selectedCycle, setSelectedCycle] = useState<string | null>(null);
  const [showBlockForm, setShowBlockForm] = useState(false);
  const [showCycleForm, setShowCycleForm] = useState(false);
  const [showInputForm, setShowInputForm] = useState(false);

  const { data: farms = [] } = useQuery({
    queryKey: ["farms"],
    queryFn: async () => (await api.get<Farm[]>("/api/v1/farms")).data,
  });

  const farmFarms = farms.filter((f) => f.type === "farm" || f.type === "greenhouse");
  const activeFarmId = selectedFarm ?? farmFarms[0]?.id ?? null;

  const { data: blocks = [] } = useQuery({
    queryKey: ["blocks", activeFarmId],
    queryFn: async () =>
      (await api.get<Block[]>("/api/v1/blocks", { params: { farm_id: activeFarmId } })).data,
    enabled: !!activeFarmId,
  });

  return (
    <div>
      <PageHeader
        title="کشت و زراعت"
        subtitle="مدیریت بلوک‌های زمین، چرخه‌های کشت و نهاده‌ها"
      />

      <div className="grid grid-cols-1 lg:grid-cols-[260px_1fr] gap-4">
        <div className="card p-3">
          <div className="text-xs font-semibold text-slate-500 mb-2 px-2">مزارع</div>
          <div className="space-y-1">
            {farmFarms.length === 0 && <div className="text-xs text-slate-400 p-3">مزرعه‌ای ثبت نشده</div>}
            {farmFarms.map((f) => (
              <button
                key={f.id}
                onClick={() => { setSelectedFarm(f.id); setSelectedCycle(null); }}
                className={`w-full text-right px-3 py-2 rounded-lg text-sm transition ${
                  activeFarmId === f.id
                    ? "bg-primary-50 text-primary-700 font-medium"
                    : "hover:bg-slate-100"
                }`}
              >
                <div className="flex items-center gap-2">
                  <MapPin className="w-3.5 h-3.5 text-slate-400" />
                  <span className="truncate">{f.name}</span>
                </div>
                {f.area_ha && (
                  <div className="text-[11px] text-slate-500 mt-0.5 mr-5">
                    {fa(f.area_ha)} هکتار
                  </div>
                )}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          {activeFarmId && (
            <BlocksSection
              farmId={activeFarmId}
              blocks={blocks}
              showForm={showBlockForm}
              onToggleForm={() => setShowBlockForm(!showBlockForm)}
            />
          )}

          {activeFarmId && (
            <CyclesSection
              blocks={blocks}
              selectedCycle={selectedCycle}
              onSelect={setSelectedCycle}
              showForm={showCycleForm}
              onToggleForm={() => setShowCycleForm(!showCycleForm)}
            />
          )}

          {selectedCycle && (
            <InputsSection
              cycleId={selectedCycle}
              showForm={showInputForm}
              onToggleForm={() => setShowInputForm(!showInputForm)}
            />
          )}

          {selectedCycle && <CostBreakdownSection cycleId={selectedCycle} />}
        </div>
      </div>
    </div>
  );
}

function BlocksSection({
  farmId, blocks, showForm, onToggleForm,
}: { farmId: string; blocks: Block[]; showForm: boolean; onToggleForm: () => void }) {
  const qc = useQueryClient();
  const { register, handleSubmit, reset } = useForm<Omit<Block, "id" | "active">>({
    defaultValues: { farm_id: farmId, name: "", area_ha: 0, soil_type: "", irrigation_type: "تیپ", notes: "" },
  });

  const mutate = useMutation({
    mutationFn: async (data: Partial<Block>) => (await api.post("/api/v1/blocks", { ...data, farm_id: farmId })).data,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["blocks"] });
      toast.success("بلوک ساخته شد");
      reset({ farm_id: farmId, name: "", area_ha: 0, soil_type: "", irrigation_type: "تیپ", notes: "" });
      onToggleForm();
    },
  });

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold flex items-center gap-2"><Sprout className="w-4 h-4 text-primary-600" /> بلوک‌های زمین</h3>
        <button onClick={onToggleForm} className="btn-secondary text-xs">
          {showForm ? <X className="w-3.5 h-3.5" /> : <Plus className="w-3.5 h-3.5" />}
          {showForm ? "بستن" : "افزودن بلوک"}
        </button>
      </div>

      {showForm && (
        <form
          onSubmit={handleSubmit((d) => mutate.mutate(d))}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4 p-3 bg-slate-50 rounded-lg"
        >
          <div>
            <label className="label">نام بلوک</label>
            <input className="input" placeholder="بلوک ۱" {...register("name", { required: true })} />
          </div>
          <div>
            <label className="label">مساحت (هکتار)</label>
            <input type="number" step="0.1" className="input" {...register("area_ha", { valueAsNumber: true, required: true })} />
          </div>
          <div>
            <label className="label">نوع خاک</label>
            <input className="input" placeholder="رسی-شنی" {...register("soil_type")} />
          </div>
          <div>
            <label className="label">آبیاری</label>
            <select className="input" {...register("irrigation_type")}>
              <option value="تیپ">تیپ</option>
              <option value="بارانی">بارانی</option>
              <option value="سطحی">سطحی</option>
            </select>
          </div>
          <div className="sm:col-span-2 lg:col-span-4 flex justify-end">
            <button type="submit" className="btn-primary" disabled={mutate.isPending}>ذخیره</button>
          </div>
        </form>
      )}

      {blocks.length === 0 ? (
        <div className="text-center text-sm text-slate-400 py-8">بلوکی برای این مزرعه ثبت نشده است</div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {blocks.map((b) => (
            <div key={b.id} className="border border-slate-200 rounded-lg p-3 hover:border-primary-300 transition">
              <div className="font-medium text-slate-800">{b.name}</div>
              <div className="text-xs text-slate-500 mt-1 space-y-0.5">
                <div>مساحت: <span className="num font-medium text-slate-700">{fa(b.area_ha)}</span> هکتار</div>
                {b.soil_type && <div>خاک: {b.soil_type}</div>}
                {b.irrigation_type && <div>آبیاری: {b.irrigation_type}</div>}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CyclesSection({
  blocks, selectedCycle, onSelect, showForm, onToggleForm,
}: { blocks: Block[]; selectedCycle: string | null; onSelect: (id: string) => void; showForm: boolean; onToggleForm: () => void }) {
  const qc = useQueryClient();
  const blockIds = new Set(blocks.map((b) => b.id));
  const { data: allCycles = [] } = useQuery({
    queryKey: ["cycles"],
    queryFn: async () => (await api.get<Cycle[]>("/api/v1/crops/cycles")).data,
  });
  const cycles = allCycles.filter((c) => blockIds.has(c.block_id));

  const { data: cropTypes = [] } = useQuery({
    queryKey: ["crop-types"],
    queryFn: async () => (await api.get<CropType[]>("/api/v1/crops/types")).data,
  });

  const { register, handleSubmit, reset } = useForm({
    defaultValues: {
      block_id: "",
      crop_type_id: "",
      season: "spring",
      year: new Date().getFullYear(),
      plant_date: "",
      target_yield_kg: 0,
    },
  });

  const mutate = useMutation({
    mutationFn: async (data: any) => (await api.post("/api/v1/crops/cycles", { ...data, plant_date: data.plant_date || null })).data,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["cycles"] });
      toast.success("چرخه کشت ساخته شد");
      reset();
      onToggleForm();
    },
  });

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold flex items-center gap-2"><Calendar className="w-4 h-4 text-primary-600" /> چرخه‌های کشت</h3>
        <button onClick={onToggleForm} className="btn-secondary text-xs" disabled={blocks.length === 0}>
          {showForm ? <X className="w-3.5 h-3.5" /> : <Plus className="w-3.5 h-3.5" />}
          {showForm ? "بستن" : "افزودن چرخه"}
        </button>
      </div>

      {showForm && (
        <form
          onSubmit={handleSubmit((d) => mutate.mutate(d))}
          className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4 p-3 bg-slate-50 rounded-lg"
        >
          <div>
            <label className="label">بلوک</label>
            <select className="input" {...register("block_id", { required: true })}>
              <option value="">— انتخاب —</option>
              {blocks.map((b) => <option key={b.id} value={b.id}>{b.name}</option>)}
            </select>
          </div>
          <div>
            <label className="label">محصول</label>
            <select className="input" {...register("crop_type_id", { required: true })}>
              <option value="">— انتخاب —</option>
              {cropTypes.map((c) => <option key={c.id} value={c.id}>{c.name_fa}</option>)}
            </select>
          </div>
          <div>
            <label className="label">فصل</label>
            <select className="input" {...register("season", { required: true })}>
              <option value="spring">بهار</option>
              <option value="summer">تابستان</option>
              <option value="fall">پاییز</option>
              <option value="winter">زمستان</option>
            </select>
          </div>
          <div>
            <label className="label">سال</label>
            <input type="number" className="input" {...register("year", { valueAsNumber: true, required: true })} />
          </div>
          <div>
            <label className="label">تاریخ کاشت</label>
            <input type="date" className="input" {...register("plant_date")} />
          </div>
          <div>
            <label className="label">هدف برداشت (کیلو)</label>
            <input type="number" className="input" {...register("target_yield_kg", { valueAsNumber: true })} />
          </div>
          <div className="sm:col-span-3 flex justify-end">
            <button type="submit" className="btn-primary" disabled={mutate.isPending}>ذخیره</button>
          </div>
        </form>
      )}

      {cycles.length === 0 ? (
        <div className="text-center text-sm text-slate-400 py-8">چرخه کشتی ثبت نشده است</div>
      ) : (
        <div className="space-y-2">
          {cycles.map((c) => (
            <button
              key={c.id}
              onClick={() => onSelect(c.id)}
              className={`w-full text-right flex items-center justify-between gap-3 p-3 rounded-lg border transition ${
                selectedCycle === c.id ? "border-primary-400 bg-primary-50" : "border-slate-200 hover:border-slate-300"
              }`}
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{c.crop_name}</span>
                  <span className="text-xs text-slate-500">— {c.block_name}</span>
                  <StatusBadge status={c.status} />
                </div>
                <div className="text-xs text-slate-500 mt-1 flex items-center gap-3 flex-wrap">
                  <span>{SEASON_LABEL[c.season] ?? c.season} <span className="num">{fa(c.year)}</span></span>
                  {c.target_yield_kg && <span>هدف: <span className="num font-medium">{fa(c.target_yield_kg)}</span> کیلو</span>}
                  {c.actual_yield_kg && <span>واقعی: <span className="num font-medium">{fa(c.actual_yield_kg)}</span> کیلو</span>}
                  {c.cost_per_kg != null && <span className="text-soil font-medium">{tomans(c.cost_per_kg)}/کیلو</span>}
                </div>
              </div>
              <ChevronRight className="w-4 h-4 text-slate-400 shrink-0 rotate-180" />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    planned: "bg-slate-100 text-slate-700",
    planting: "bg-sky-100 text-sky-700",
    growing: "bg-emerald-100 text-emerald-700",
    harvested: "bg-amber-100 text-amber-700",
    closed: "bg-slate-200 text-slate-500",
  };
  return (
    <span className={`text-[10px] px-2 py-0.5 rounded-full ${colors[status] ?? "bg-slate-100 text-slate-700"}`}>
      {STATUS_LABEL[status] ?? status}
    </span>
  );
}

interface Input {
  id: string;
  cycle_id: string;
  input_category: string;
  input_type: string;
  date: string;
  quantity: number;
  unit: string;
  unit_price: number;
  total_price: number;
}

const CATEGORY_LABEL: Record<string, string> = {
  seed: "بذر/نشا",
  fertilizer: "کود",
  pesticide: "سم",
  water: "آب",
  labor: "کارگر",
  machinery: "ماشین‌آلات",
  overhead: "سربار",
  transport: "حمل",
  waste: "ضایعات",
  electricity: "برق",
  fuel: "سوخت",
};

function InputsSection({ cycleId, showForm, onToggleForm }: { cycleId: string; showForm: boolean; onToggleForm: () => void }) {
  const qc = useQueryClient();
  const { data: inputs = [] } = useQuery({
    queryKey: ["cycle-inputs", cycleId],
    queryFn: async () => (await api.get<Input[]>(`/api/v1/crops/cycles/${cycleId}/inputs`)).data,
  });

  const { register, handleSubmit, reset, watch } = useForm({
    defaultValues: {
      input_category: "fertilizer",
      input_type: "",
      date: new Date().toISOString().slice(0, 10),
      quantity: 0,
      unit: "kg",
      unit_price: 0,
      supplier: "",
    },
  });

  const qty = watch("quantity") || 0;
  const price = watch("unit_price") || 0;

  const mutate = useMutation({
    mutationFn: async (data: any) => (await api.post(`/api/v1/crops/cycles/${cycleId}/inputs`, data)).data,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["cycle-inputs", cycleId] });
      qc.invalidateQueries({ queryKey: ["cycle-cost", cycleId] });
      qc.invalidateQueries({ queryKey: ["cycles"] });
      toast.success("نهاده ثبت شد و قیمت تمام‌شده به‌روز شد");
      reset({ input_category: "fertilizer", input_type: "", date: new Date().toISOString().slice(0, 10), quantity: 0, unit: "kg", unit_price: 0, supplier: "" });
    },
  });

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold flex items-center gap-2"><Tractor className="w-4 h-4 text-primary-600" /> نهاده‌های ثبت‌شده</h3>
        <button onClick={onToggleForm} className="btn-secondary text-xs">
          {showForm ? <X className="w-3.5 h-3.5" /> : <Plus className="w-3.5 h-3.5" />}
          {showForm ? "بستن" : "افزودن نهاده"}
        </button>
      </div>

      {showForm && (
        <form
          onSubmit={handleSubmit((d) => mutate.mutate(d))}
          className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4 p-3 bg-slate-50 rounded-lg"
        >
          <div>
            <label className="label">دسته</label>
            <select className="input" {...register("input_category")}>
              <option value="seed">بذر/نشا</option>
              <option value="fertilizer">کود</option>
              <option value="pesticide">سم</option>
              <option value="water">آب</option>
              <option value="overhead">سربار</option>
              <option value="transport">حمل</option>
              <option value="waste">ضایعات</option>
            </select>
          </div>
          <div>
            <label className="label">نوع/نام</label>
            <input className="input" placeholder="کود اوره" {...register("input_type", { required: true })} />
          </div>
          <div>
            <label className="label">تاریخ</label>
            <input type="date" className="input" {...register("date", { required: true })} />
          </div>
          <div>
            <label className="label">واحد</label>
            <select className="input" {...register("unit")}>
              <option value="kg">کیلوگرم</option>
              <option value="L">لیتر</option>
              <option value="m³">متر مکعب</option>
              <option value="each">عدد</option>
              <option value="نفر-روز">نفر-روز</option>
              <option value="ساعت">ساعت</option>
            </select>
          </div>
          <div>
            <label className="label">مقدار</label>
            <input type="number" step="0.001" className="input" {...register("quantity", { valueAsNumber: true, required: true })} />
          </div>
          <div>
            <label className="label">قیمت واحد (تومان)</label>
            <input type="number" className="input" {...register("unit_price", { valueAsNumber: true, required: true })} />
          </div>
          <div>
            <label className="label">جمع</label>
            <input className="input bg-slate-100" disabled value={fa(qty * price)} />
          </div>
          <div>
            <label className="label">تأمین‌کننده</label>
            <input className="input" {...register("supplier")} />
          </div>
          <div className="col-span-2 lg:col-span-4 flex justify-end">
            <button type="submit" className="btn-primary" disabled={mutate.isPending}>ذخیره</button>
          </div>
        </form>
      )}

      {inputs.length === 0 ? (
        <div className="text-center text-sm text-slate-400 py-8">نهاده‌ای ثبت نشده — هزینه‌ها را اضافه کنید تا قیمت تمام‌شده محاسبه شود</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-xs text-slate-500 border-b border-slate-200">
              <tr>
                <th className="text-right py-2 px-2">تاریخ</th>
                <th className="text-right py-2 px-2">دسته</th>
                <th className="text-right py-2 px-2">نام</th>
                <th className="text-right py-2 px-2">مقدار</th>
                <th className="text-right py-2 px-2">قیمت واحد</th>
                <th className="text-right py-2 px-2">جمع</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {inputs.map((i) => (
                <tr key={i.id}>
                  <td className="py-2 px-2 num text-slate-600">{i.date}</td>
                  <td className="py-2 px-2"><span className="text-xs bg-slate-100 px-2 py-0.5 rounded">{CATEGORY_LABEL[i.input_category] ?? i.input_category}</span></td>
                  <td className="py-2 px-2">{i.input_type}</td>
                  <td className="py-2 px-2 num">{fa(i.quantity)} {i.unit}</td>
                  <td className="py-2 px-2 num">{tomans(i.unit_price)}</td>
                  <td className="py-2 px-2 num font-medium">{tomans(i.total_price)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

interface CostResult {
  cycle_id: string;
  crop: string | null;
  yield: { target_kg: number | null; actual_kg: number };
  breakdown: {
    direct_materials: number;
    direct_labor: number;
    machinery: number;
    utilities: number;
    overhead: number;
    total: number;
    by_subcategory: Record<string, number>;
  };
  summary: {
    total_cost: number;
    cost_per_kg: number | null;
    market_price_per_kg: number | null;
    savings_vs_market_pct: number | null;
  };
}

function CostBreakdownSection({ cycleId }: { cycleId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ["cycle-cost", cycleId],
    queryFn: async () => (await api.get<CostResult>(`/api/v1/crops/cycles/${cycleId}/cost`)).data,
  });

  if (isLoading || !data) return <div className="card p-6 text-center text-sm text-slate-400">در حال محاسبه قیمت تمام‌شده...</div>;

  const b = data.breakdown;
  const total = b.total || 1;

  const buckets = [
    { key: "direct_materials", label: "نهاده‌های مستقیم", color: "bg-primary-500", amount: b.direct_materials },
    { key: "direct_labor", label: "کارگر", color: "bg-amber-500", amount: b.direct_labor },
    { key: "machinery", label: "ماشین‌آلات", color: "bg-soil", amount: b.machinery },
    { key: "utilities", label: "آب و انرژی", color: "bg-water", amount: b.utilities },
    { key: "overhead", label: "سربار و سایر", color: "bg-slate-500", amount: b.overhead },
  ];

  return (
    <div className="card p-4">
      <h3 className="font-semibold mb-1">قیمت تمام‌شده — {data.crop ?? "—"}</h3>
      <p className="text-xs text-slate-500 mb-4">محاسبه خودکار از روی نهاده‌ها، کارگر و ماشین‌آلات ثبت‌شده</p>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
        <div className="card p-3 border-t-4 border-soil">
          <div className="text-xs text-slate-500">جمع هزینه</div>
          <div className="text-xl font-bold mt-1 num">{tomans(b.total)}</div>
        </div>
        <div className="card p-3 border-t-4 border-primary-600">
          <div className="text-xs text-slate-500">قیمت تمام‌شده هر کیلو</div>
          <div className="text-xl font-bold mt-1 num">
            {data.summary.cost_per_kg ? tomans(data.summary.cost_per_kg) : "—"}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">
            {data.yield.actual_kg > 0 ? `از ${fa(data.yield.actual_kg)} کیلو برداشت` : "هنوز برداشت ثبت نشده"}
          </div>
        </div>
        <div className="card p-3 border-t-4 border-water">
          <div className="text-xs text-slate-500">قیمت بازار</div>
          <div className="text-xl font-bold mt-1 num">
            {data.summary.market_price_per_kg ? tomans(data.summary.market_price_per_kg) : "—"}
          </div>
          {data.summary.savings_vs_market_pct != null && (
            <div className={`text-[11px] mt-0.5 font-medium ${data.summary.savings_vs_market_pct >= 0 ? "text-emerald-600" : "text-red-600"}`}>
              {data.summary.savings_vs_market_pct >= 0 ? "صرفه‌جویی" : "زیان"}: {fa(Math.abs(data.summary.savings_vs_market_pct))}٪
            </div>
          )}
        </div>
      </div>

      <div className="space-y-2">
        {buckets.filter((x) => x.amount > 0).map((x) => {
          const pct = (x.amount / total) * 100;
          return (
            <div key={x.key}>
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="font-medium text-slate-700">{x.label}</span>
                <span className="num text-slate-600">{tomans(x.amount)} <span className="text-slate-400">({fa(Math.round(pct))}٪)</span></span>
              </div>
              <div className="h-2 bg-slate-100 rounded overflow-hidden">
                <div className={`h-full ${x.color}`} style={{ width: `${pct}%` }} />
              </div>
            </div>
          );
        })}
        {buckets.every((x) => x.amount === 0) && (
          <div className="text-center text-sm text-slate-400 py-4">هنوز هزینه‌ای ثبت نشده است</div>
        )}
      </div>
    </div>
  );
}
