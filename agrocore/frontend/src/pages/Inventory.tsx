import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import { Plus, X, AlertTriangle, ArrowDown, ArrowUp, Package } from "lucide-react";

import { api } from "@/lib/api";
import { fa, tomans } from "@/lib/utils";
import { PageHeader } from "@/components/PageHeader";

interface Item {
  id: string;
  name: string;
  category: string;
  unit: string;
  current_qty: number;
  min_qty: number;
  last_unit_price: number | null;
  notes: string | null;
  low_stock: boolean;
}

interface Tx {
  id: string;
  item_id: string;
  direction: "in" | "out";
  source: string;
  date: string;
  quantity: number;
  unit_price: number | null;
  total_price: number | null;
  batch_code: string | null;
  expiry_date: string | null;
  notes: string | null;
}

const CATEGORY_LABEL: Record<string, string> = {
  raw_material: "ماده اولیه",
  processed: "محصول فراوری‌شده",
  feed: "خوراک دام",
  consumable: "مصرفی",
  packaging: "بسته‌بندی",
};

const SOURCE_LABEL: Record<string, string> = {
  farm: "مزرعه",
  livestock: "دامداری",
  processing: "فراوری",
  purchase: "خرید",
  kitchen: "آشپزخانه",
  sale: "فروش",
  waste: "ضایعات",
};

export function InventoryPage() {
  const [showCreate, setShowCreate] = useState(false);
  const [lowOnly, setLowOnly] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  const { data: items = [] } = useQuery({
    queryKey: ["inventory-items", lowOnly],
    queryFn: async () => (await api.get<Item[]>("/api/v1/inventory/items", { params: { low_only: lowOnly } })).data,
  });

  return (
    <div>
      <PageHeader
        title="انبار یکپارچه"
        subtitle="موجودی، ورود از مزرعه/فراوری، خروج به آشپزخانه"
        actions={
          <>
            <button
              onClick={() => setLowOnly(!lowOnly)}
              className={lowOnly ? "btn-primary" : "btn-secondary"}
            >
              <AlertTriangle className="w-4 h-4" />
              {lowOnly ? "همه" : "فقط موجودی کم"}
            </button>
            <button onClick={() => setShowCreate(!showCreate)} className="btn-primary">
              {showCreate ? <X className="w-4 h-4" /> : <Plus className="w-4 h-4" />}
              {showCreate ? "بستن" : "کالای جدید"}
            </button>
          </>
        }
      />

      {showCreate && <CreateItemForm onClose={() => setShowCreate(false)} />}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {items.length === 0 ? (
          <div className="card p-6 lg:col-span-3 text-center text-sm text-slate-400">
            {lowOnly ? "هیچ موجودی زیر حد آستانه نیست — انبار سالم است" : "هیچ کالایی ثبت نشده"}
          </div>
        ) : (
          items.map((item) => (
            <ItemCard
              key={item.id}
              item={item}
              expanded={selected === item.id}
              onToggle={() => setSelected(selected === item.id ? null : item.id)}
            />
          ))
        )}
      </div>
    </div>
  );
}

function ItemCard({ item, expanded, onToggle }: { item: Item; expanded: boolean; onToggle: () => void }) {
  const ratio = item.min_qty > 0 ? Math.min(100, (item.current_qty / item.min_qty) * 100) : 100;
  const ratioColor = item.low_stock ? "bg-red-500" : ratio > 200 ? "bg-emerald-500" : "bg-primary-500";

  return (
    <div className={`card overflow-hidden ${expanded ? "lg:col-span-3" : ""}`}>
      <button onClick={onToggle} className="w-full text-right p-4 hover:bg-slate-50 transition">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-start gap-3 min-w-0">
            <div className="w-10 h-10 rounded-lg bg-slate-100 grid place-items-center shrink-0">
              <Package className="w-5 h-5 text-slate-500" />
            </div>
            <div className="min-w-0">
              <div className="font-medium truncate">{item.name}</div>
              <div className="text-xs text-slate-500 mt-0.5">
                {CATEGORY_LABEL[item.category] ?? item.category}
              </div>
            </div>
          </div>
          {item.low_stock && (
            <span className="text-[10px] bg-red-100 text-red-700 px-2 py-0.5 rounded-full whitespace-nowrap">
              کمبود
            </span>
          )}
        </div>

        <div className="mt-3">
          <div className="flex items-end justify-between mb-1">
            <span className="text-xl font-bold num">{fa(item.current_qty)} <span className="text-xs font-normal text-slate-500">{item.unit}</span></span>
            <span className="text-[11px] text-slate-400">حداقل: <span className="num">{fa(item.min_qty)}</span></span>
          </div>
          <div className="h-1.5 bg-slate-100 rounded overflow-hidden">
            <div className={`h-full ${ratioColor}`} style={{ width: `${ratio}%` }} />
          </div>
          {item.last_unit_price && (
            <div className="text-[11px] text-slate-500 mt-1.5 num">
              آخرین قیمت: {tomans(item.last_unit_price)}/{item.unit}
            </div>
          )}
        </div>
      </button>

      {expanded && <TransactionsSection itemId={item.id} unit={item.unit} />}
    </div>
  );
}

function CreateItemForm({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const { register, handleSubmit, reset } = useForm({
    defaultValues: { name: "", category: "raw_material", unit: "kg", current_qty: 0, min_qty: 0, last_unit_price: 0 },
  });

  const mutate = useMutation({
    mutationFn: async (d: any) => (await api.post("/api/v1/inventory/items", d)).data,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["inventory-items"] });
      toast.success("کالا ساخته شد");
      reset();
      onClose();
    },
  });

  return (
    <form
      onSubmit={handleSubmit((d) => mutate.mutate(d))}
      className="card p-4 mb-4 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3"
    >
      <div className="col-span-2">
        <label className="label">نام کالا</label>
        <input className="input" {...register("name", { required: true })} />
      </div>
      <div>
        <label className="label">دسته</label>
        <select className="input" {...register("category")}>
          {Object.entries(CATEGORY_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
      </div>
      <div>
        <label className="label">واحد</label>
        <select className="input" {...register("unit")}>
          <option value="kg">کیلوگرم</option>
          <option value="L">لیتر</option>
          <option value="each">عدد</option>
          <option value="m³">متر مکعب</option>
        </select>
      </div>
      <div>
        <label className="label">حداقل موجودی</label>
        <input type="number" className="input" {...register("min_qty", { valueAsNumber: true })} />
      </div>
      <div>
        <label className="label">قیمت واحد</label>
        <input type="number" className="input" {...register("last_unit_price", { valueAsNumber: true })} />
      </div>
      <div className="col-span-2 sm:col-span-3 lg:col-span-6 flex justify-end gap-2">
        <button type="button" onClick={onClose} className="btn-secondary">انصراف</button>
        <button type="submit" className="btn-primary" disabled={mutate.isPending}>ذخیره</button>
      </div>
    </form>
  );
}

function TransactionsSection({ itemId, unit }: { itemId: string; unit: string }) {
  const qc = useQueryClient();
  const { data: txs = [] } = useQuery({
    queryKey: ["inventory-tx", itemId],
    queryFn: async () => (await api.get<Tx[]>(`/api/v1/inventory/items/${itemId}/transactions`)).data,
  });

  const { register, handleSubmit, reset } = useForm({
    defaultValues: {
      direction: "in" as "in" | "out",
      source: "purchase",
      date: new Date().toISOString().slice(0, 10),
      quantity: 0,
      unit_price: 0,
      batch_code: "",
      notes: "",
    },
  });

  const mutate = useMutation({
    mutationFn: async (d: any) => (await api.post(`/api/v1/inventory/items/${itemId}/transactions`, d)).data,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["inventory-tx", itemId] });
      qc.invalidateQueries({ queryKey: ["inventory-items"] });
      toast.success("تراکنش ثبت شد");
      reset({ direction: "in", source: "purchase", date: new Date().toISOString().slice(0, 10), quantity: 0, unit_price: 0, batch_code: "", notes: "" });
    },
  });

  return (
    <div className="border-t border-slate-200 p-4 bg-slate-50">
      <form
        onSubmit={handleSubmit((d) => mutate.mutate(d))}
        className="grid grid-cols-2 lg:grid-cols-7 gap-2 mb-4"
      >
        <select className="input" {...register("direction")}>
          <option value="in">ورود</option>
          <option value="out">خروج</option>
        </select>
        <select className="input" {...register("source")}>
          {Object.entries(SOURCE_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
        <input type="date" className="input" {...register("date")} />
        <input type="number" step="0.001" placeholder="مقدار" className="input" {...register("quantity", { valueAsNumber: true })} />
        <input type="number" placeholder="قیمت واحد" className="input" {...register("unit_price", { valueAsNumber: true })} />
        <input placeholder="کد بچ" className="input" {...register("batch_code")} />
        <button type="submit" className="btn-primary" disabled={mutate.isPending}>ثبت</button>
      </form>

      {txs.length === 0 ? (
        <div className="text-sm text-slate-400 text-center py-4">تراکنشی ثبت نشده</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-xs text-slate-500 border-b border-slate-300">
              <tr>
                <th className="text-right py-2 px-2">تاریخ</th>
                <th className="text-right py-2 px-2">جهت</th>
                <th className="text-right py-2 px-2">منبع</th>
                <th className="text-right py-2 px-2">مقدار</th>
                <th className="text-right py-2 px-2">قیمت واحد</th>
                <th className="text-right py-2 px-2">جمع</th>
                <th className="text-right py-2 px-2">بچ</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {txs.map((t) => (
                <tr key={t.id}>
                  <td className="py-1.5 px-2 num text-slate-600">{t.date}</td>
                  <td className="py-1.5 px-2">
                    {t.direction === "in" ? (
                      <span className="inline-flex items-center gap-1 text-emerald-700 text-xs"><ArrowDown className="w-3 h-3" />ورود</span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-red-700 text-xs"><ArrowUp className="w-3 h-3" />خروج</span>
                    )}
                  </td>
                  <td className="py-1.5 px-2 text-xs">{SOURCE_LABEL[t.source] ?? t.source}</td>
                  <td className="py-1.5 px-2 num">{fa(t.quantity)} {unit}</td>
                  <td className="py-1.5 px-2 num">{t.unit_price ? tomans(t.unit_price) : "—"}</td>
                  <td className="py-1.5 px-2 num font-medium">{t.total_price ? tomans(t.total_price) : "—"}</td>
                  <td className="py-1.5 px-2 text-xs font-mono">{t.batch_code ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
