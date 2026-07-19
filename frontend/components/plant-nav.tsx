"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboardIcon,
  MapPinIcon,
  DatabaseIcon,
  FlaskConicalIcon,
  TrendingUpIcon,
  ActivityIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface NavItem {
  label: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  exact?: boolean;
}

interface Props {
  plantId: string;
}

export default function PlantNav({ plantId }: Props) {
  const pathname = usePathname();

  const items: NavItem[] = [
    {
      label: "Genel Bakış",
      href: `/plants/${plantId}`,
      icon: LayoutDashboardIcon,
      exact: true,
    },
    {
      label: "Harita & Noktalar",
      href: `/plants/${plantId}/map`,
      icon: MapPinIcon,
    },
    {
      label: "Geçmiş Veri",
      href: `/plants/${plantId}/data`,
      icon: DatabaseIcon,
    },
    {
      label: "Experiment'ler",
      href: `/plants/${plantId}/experiments`,
      icon: FlaskConicalIcon,
    },
    {
      label: "Forecast'ler",
      href: `/plants/${plantId}/forecasts`,
      icon: TrendingUpIcon,
    },
    {
      label: "Drift & İzleme",
      href: `/plants/${plantId}/monitoring`,
      icon: ActivityIcon,
    },
  ];

  return (
    <nav className="flex gap-1 overflow-x-auto border-b pb-0 -mb-px">
      {items.map((item) => {
        const isActive = item.exact
          ? pathname === item.href
          : pathname.startsWith(item.href);

        return (
          <Link
            key={item.href}
            href={item.href}
            className={cn(
              "flex items-center gap-1.5 whitespace-nowrap border-b-2 px-3 py-2 text-sm font-medium transition-colors",
              isActive
                ? "border-primary text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground"
            )}
          >
            <item.icon className="size-4" />
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
