import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const numberFormatter = new Intl.NumberFormat("fa-IR");

export function fa(n: number | bigint | string | null | undefined): string {
  if (n === null || n === undefined || n === "") return "—";
  const num = typeof n === "string" ? Number(n) : n;
  if (typeof num === "number" && Number.isNaN(num)) return "—";
  return numberFormatter.format(num as number);
}

export function tomans(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  return `${fa(Math.round(n))} تومان`;
}
