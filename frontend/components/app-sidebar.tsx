import Link from "next/link";
import {
  LayoutDashboardIcon,
  MapPinIcon,
  DatabaseIcon,
  FlaskConicalIcon,
  TrendingUpIcon,
  BuildingIcon,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";
import PlantSwitcher from "@/components/plant-switcher";
import { listPlants } from "@/lib/api";
import type { Plant } from "@/lib/types";

interface PlantNavItem {
  label: string;
  href: (plantId: string) => string;
  icon: React.ComponentType<{ className?: string }>;
}

const plantNavItems: PlantNavItem[] = [
  { label: "Genel Bakış", href: (id) => `/plants/${id}`, icon: LayoutDashboardIcon },
  { label: "Harita & Noktalar", href: (id) => `/plants/${id}/map`, icon: MapPinIcon },
  { label: "Geçmiş Veri", href: (id) => `/plants/${id}/data`, icon: DatabaseIcon },
  { label: "Experiment'ler", href: (id) => `/plants/${id}/experiments`, icon: FlaskConicalIcon },
  { label: "Forecast'ler", href: (id) => `/plants/${id}/forecasts`, icon: TrendingUpIcon },
];

interface AppSidebarProps {
  activePlantId?: string;
}

export default async function AppSidebar({ activePlantId }: AppSidebarProps) {
  let plants: Plant[] = [];
  try {
    plants = await listPlants();
  } catch {
    // Backend might be offline during build; sidebar renders with empty list
    plants = [];
  }

  return (
    <Sidebar variant="inset">
      <SidebarHeader>
        <div className="flex items-center gap-2 px-2 py-1">
          <div className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <TrendingUpIcon className="size-4" />
          </div>
          <div className="grid flex-1 text-left text-sm leading-tight">
            <span className="font-semibold">OpenEnergy</span>
            <span className="text-xs text-muted-foreground">Tahmin Platformu</span>
          </div>
        </div>
        <Separator />
        <PlantSwitcher plants={plants} />
      </SidebarHeader>

      <SidebarContent>
        {activePlantId ? (
          <SidebarGroup>
            <SidebarGroupLabel>Santral</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {plantNavItems.map((item) => (
                  <SidebarMenuItem key={item.label}>
                    <SidebarMenuButton asChild>
                      <Link href={item.href(activePlantId)}>
                        <item.icon className="size-4" />
                        <span>{item.label}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ) : (
          <SidebarGroup>
            <SidebarGroupContent>
              <SidebarMenu>
                <SidebarMenuItem>
                  <SidebarMenuButton asChild>
                    <Link href="/plants">
                      <BuildingIcon className="size-4" />
                      <span>Santrallerim</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        )}
      </SidebarContent>

      <SidebarFooter>
        <div className="px-2 py-2 text-xs text-muted-foreground">
          OpenEnergy v0.1
        </div>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
