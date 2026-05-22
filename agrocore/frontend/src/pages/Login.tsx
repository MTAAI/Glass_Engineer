import { useForm } from "react-hook-form";
import { useNavigate, useLocation, Navigate } from "react-router-dom";
import { useState } from "react";
import { toast } from "sonner";
import { z } from "zod";

import { api, type TokenResponse } from "@/lib/api";
import { useAuthStore } from "@/store/auth";

const LoginSchema = z.object({
  email: z.string().email("ایمیل معتبر وارد کنید"),
  password: z.string().min(4, "حداقل ۴ کاراکتر"),
});

type LoginForm = z.infer<typeof LoginSchema>;

const DEMO_ACCOUNTS = [
  { email: "admin@agrocore.local", password: "admin1234", role: "مدیر کل" },
  { email: "manager@agrocore.local", password: "manager1234", role: "مدیر اجرایی" },
  { email: "farm@agrocore.local", password: "farm1234", role: "مدیر مزرعه" },
  { email: "livestock@agrocore.local", password: "livestock1234", role: "مدیر دامداری" },
  { email: "kitchen@agrocore.local", password: "kitchen1234", role: "سرآشپز" },
];

export function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const setSession = useAuthStore((s) => s.setSession);
  const token = useAuthStore((s) => s.token);
  const [submitting, setSubmitting] = useState(false);

  const {
    register,
    handleSubmit,
    setValue,
    formState: { errors },
  } = useForm<LoginForm>({ defaultValues: { email: "", password: "" } });

  if (token) {
    return <Navigate to="/" replace />;
  }

  async function onSubmit(values: LoginForm) {
    const parsed = LoginSchema.safeParse(values);
    if (!parsed.success) {
      toast.error(parsed.error.errors[0]?.message ?? "خطای اعتبارسنجی");
      return;
    }
    setSubmitting(true);
    try {
      const r = await api.post<TokenResponse>("/api/v1/auth/login", parsed.data);
      const { access_token, ...user } = r.data;
      setSession(access_token, user);
      toast.success(`خوش آمدید ${user.name}`);
      const from = (location.state as { from?: { pathname: string } } | null)?.from?.pathname ?? "/";
      navigate(from, { replace: true });
    } catch {
      // toast handled by interceptor
    } finally {
      setSubmitting(false);
    }
  }

  function fillDemo(email: string, password: string) {
    setValue("email", email);
    setValue("password", password);
  }

  return (
    <div className="min-h-screen grid lg:grid-cols-2 bg-gradient-to-br from-primary-50 to-emerald-50">
      <div className="hidden lg:flex flex-col justify-center px-12 py-12 bg-primary-600 text-white relative overflow-hidden">
        <div className="absolute inset-0 opacity-10 text-[20rem] leading-none select-none pointer-events-none">🌿</div>
        <div className="relative z-10">
          <div className="text-4xl font-bold mb-4">AgroCore OS</div>
          <div className="text-xl mb-2 text-primary-100">سامانه جامع مدیریت کشت و صنعت</div>
          <div className="text-sm text-primary-100/80">کارخانه‌جات شیشه قزوین — نسخه ۱.۰</div>

          <div className="mt-12 space-y-3 text-sm text-primary-50">
            <div className="flex items-start gap-2">
              <span>✓</span>
              <span>ردیابی هزینه از خرید بذر تا سفره</span>
            </div>
            <div className="flex items-start gap-2">
              <span>✓</span>
              <span>محاسبه قیمت تمام‌شده واقعی هر محصول</span>
            </div>
            <div className="flex items-start gap-2">
              <span>✓</span>
              <span>یکپارچگی مزرعه + دامداری + فراوری + آشپزخانه</span>
            </div>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-md">
          <div className="lg:hidden text-center mb-6">
            <div className="text-3xl font-bold text-primary-700">🌿 AgroCore OS</div>
            <div className="text-sm text-slate-500 mt-1">سامانه کشت‌وصنعت</div>
          </div>

          <div className="card p-6">
            <h2 className="text-xl font-bold mb-1">ورود به سامانه</h2>
            <p className="text-sm text-slate-500 mb-5">با ایمیل و رمز خود وارد شوید</p>

            <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
              <div>
                <label className="label">ایمیل</label>
                <input
                  type="email"
                  className="input"
                  autoComplete="email"
                  dir="ltr"
                  {...register("email")}
                />
                {errors.email && <p className="text-xs text-red-600 mt-1">{errors.email.message}</p>}
              </div>

              <div>
                <label className="label">رمز عبور</label>
                <input
                  type="password"
                  className="input"
                  autoComplete="current-password"
                  dir="ltr"
                  {...register("password")}
                />
                {errors.password && <p className="text-xs text-red-600 mt-1">{errors.password.message}</p>}
              </div>

              <button type="submit" className="btn-primary w-full" disabled={submitting}>
                {submitting ? "در حال ورود..." : "ورود"}
              </button>
            </form>

            <div className="mt-6 pt-5 border-t border-slate-200">
              <div className="text-xs font-medium text-slate-500 mb-2">حساب‌های دموی:</div>
              <div className="space-y-1 text-xs">
                {DEMO_ACCOUNTS.map((a) => (
                  <button
                    key={a.email}
                    type="button"
                    onClick={() => fillDemo(a.email, a.password)}
                    className="w-full flex items-center justify-between gap-2 px-2 py-1.5 rounded hover:bg-slate-100 text-right"
                  >
                    <span className="font-medium text-slate-700">{a.role}</span>
                    <span className="text-slate-400 font-mono text-[10px]" dir="ltr">{a.email}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
