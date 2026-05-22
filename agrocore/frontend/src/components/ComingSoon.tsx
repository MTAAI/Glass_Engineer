import { Construction } from "lucide-react";

import { PageHeader } from "./PageHeader";

interface Props {
  title: string;
  subtitle?: string;
  phase: string;
  features: string[];
}

export function ComingSoon({ title, subtitle, phase, features }: Props) {
  return (
    <div>
      <PageHeader title={title} subtitle={subtitle} />

      <div className="card p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-12 h-12 rounded-lg bg-amber-100 text-amber-700 grid place-items-center">
            <Construction className="w-6 h-6" />
          </div>
          <div>
            <div className="font-semibold text-slate-900">در مرحله {phase} پیاده‌سازی می‌شود</div>
            <div className="text-sm text-slate-500">این بخش بخشی از نقشه راه توسعه AgroCore OS است</div>
          </div>
        </div>

        <div className="mt-4">
          <div className="text-sm font-medium text-slate-700 mb-2">قابلیت‌های پیش‌بینی‌شده:</div>
          <ul className="space-y-1.5">
            {features.map((f, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-slate-600">
                <span className="text-primary-600 mt-1">•</span>
                <span>{f}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
