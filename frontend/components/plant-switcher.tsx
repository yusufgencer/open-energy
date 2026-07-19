"use client";

import { useRouter, useParams } from "next/navigation";
import { ChevronsUpDownIcon, WindIcon, SunIcon, PlusIcon } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenu,
} from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import type { Plant } from "@/lib/types";

interface Props {
  plants: Plant[];
}

export default function PlantSwitcher({ plants }: Props) {
  const router = useRouter();
  const params = useParams();
  const currentId = params?.plantId as string | undefined;
  const current = plants.find((p) => p.plant_id === currentId) ?? plants[0];

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <SidebarMenuButton
              size="lg"
              className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
            >
              <div
                className={cn(
                  "flex size-8 items-center justify-center rounded-lg text-white",
                  current?.kind === "solar" ? "bg-amber-400" : "bg-sky-500"
                )}
              >
                {current?.kind === "solar" ? (
                  <SunIcon className="size-4" />
                ) : (
                  <WindIcon className="size-4" />
                )}
              </div>
              <div className="grid flex-1 text-left text-sm leading-tight">
                <span className="truncate font-semibold">
                  {current?.name ?? "Santral seç"}
                </span>
                <span className="truncate text-xs text-muted-foreground">
                  {current ? `${current.capacity_mw} MW · ${current.kind}` : "—"}
                </span>
              </div>
              <ChevronsUpDownIcon className="ml-auto size-4" />
            </SidebarMenuButton>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            className="w-[--radix-dropdown-menu-trigger-width] min-w-56 rounded-lg"
            align="start"
            side="bottom"
            sideOffset={4}
          >
            {plants.map((p) => (
              <DropdownMenuItem
                key={p.plant_id}
                onClick={() => router.push(`/plants/${p.plant_id}`)}
                className="gap-2 p-2"
              >
                <div
                  className={cn(
                    "flex size-6 items-center justify-center rounded-sm text-white",
                    p.kind === "solar" ? "bg-amber-400" : "bg-sky-500"
                  )}
                >
                  {p.kind === "solar" ? (
                    <SunIcon className="size-3" />
                  ) : (
                    <WindIcon className="size-3" />
                  )}
                </div>
                <span className="flex-1 truncate">{p.name}</span>
              </DropdownMenuItem>
            ))}
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className="gap-2 p-2"
              onClick={() => router.push("/plants")}
            >
              <div className="flex size-6 items-center justify-center rounded-md border">
                <PlusIcon className="size-4" />
              </div>
              <span className="text-muted-foreground">Yeni santral</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </SidebarMenuItem>
    </SidebarMenu>
  );
}
