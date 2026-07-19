"use client";

import dynamic from "next/dynamic";
import type { PlantPoint } from "@/lib/types";

const MapPanel = dynamic(() => import("@/components/map-panel"), { ssr: false });

interface Props {
  plantId: string;
  initialPoints: PlantPoint[];
}

export default function MapPanelWrapper({ plantId, initialPoints }: Props) {
  return <MapPanel plantId={plantId} initialPoints={initialPoints} />;
}
