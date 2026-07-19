"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { toast } from "sonner";
import { CloudIcon, WindIcon, SunIcon } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { addPoint } from "@/lib/api";
import type { PlantPoint, PointType } from "@/lib/types";

const POINT_COLORS: Record<PointType, string> = {
  weather: "#0EA5E9",   // sky-500
  turbine: "#0F766E",   // teal-700
  panel_array: "#F59E0B", // amber-400
};

const POINT_LABELS: Record<PointType, string> = {
  weather: "Hava",
  turbine: "Türbin",
  panel_array: "Panel",
};

const POINT_ICONS: Record<PointType, React.ComponentType<{ className?: string }>> = {
  weather: CloudIcon,
  turbine: WindIcon,
  panel_array: SunIcon,
};

interface Props {
  plantId: string;
  initialPoints: PlantPoint[];
}

export default function MapPanel({ plantId, initialPoints }: Props) {
  const mapRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const mapInstance = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const markersRef = useRef<any[]>([]);
  const resizeObsRef = useRef<ResizeObserver | null>(null);
  const [points, setPoints] = useState<PlantPoint[]>(initialPoints);
  const [selectedType, setSelectedType] = useState<PointType>("weather");
  const selectedTypeRef = useRef<PointType>(selectedType);
  const [addMode, setAddMode] = useState(false);
  const addModeRef = useRef(false);
  const [mapLoaded, setMapLoaded] = useState(false);

  useEffect(() => {
    selectedTypeRef.current = selectedType;
  }, [selectedType]);

  useEffect(() => {
    addModeRef.current = addMode;
  }, [addMode]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const addMarker = useCallback((maplibregl: any, map: any, pt: PlantPoint) => {
    const el = document.createElement("div");
    const color = POINT_COLORS[pt.point_type];
    el.style.cssText = `
      width: 24px; height: 24px; border-radius: 50%;
      background: ${color}; border: 2px solid white;
      box-shadow: 0 2px 4px rgba(0,0,0,0.3); cursor: pointer;
    `;
    const marker = new maplibregl.Marker({ element: el })
      .setLngLat([pt.longitude, pt.latitude])
      .setPopup(
        new maplibregl.Popup({ offset: 16 }).setHTML(
          `<p style="margin:0;font-size:12px;font-weight:600">${POINT_LABELS[pt.point_type]}</p>
           <p style="margin:0;font-size:11px;color:#6b7280">${pt.latitude.toFixed(4)}, ${pt.longitude.toFixed(4)}</p>`
        )
      )
      .addTo(map);
    markersRef.current.push(marker);
    return marker;
  }, []);

  useEffect(() => {
    if (!mapRef.current || mapInstance.current) return;

    import("maplibre-gl").then(({ default: maplibregl }) => {
      if (!mapRef.current) return;

      const map = new maplibregl.Map({
        container: mapRef.current,
        style: {
          version: 8,
          sources: {
            osm: {
              type: "raster",
              tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
              tileSize: 256,
              attribution: "© OpenStreetMap contributors",
            },
          },
          layers: [{ id: "osm", type: "raster", source: "osm" }],
        },
        center: [35.0, 39.0],
        zoom: 5,
      });

      mapInstance.current = map;

      // MapLibre dinamik import nedeniyle container layout'tan önce init olabilir
      // (0 yükseklik) ve otomatik resize etmez → ResizeObserver ile düzelt.
      const ro = new ResizeObserver(() => map.resize());
      if (mapRef.current) ro.observe(mapRef.current);
      resizeObsRef.current = ro;

      map.on("load", () => {
        setMapLoaded(true);
        map.resize();
        // Add existing markers
        points.forEach((pt) => addMarker(maplibregl, map, pt));
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      map.on("click", async (e: any) => {
        if (!addModeRef.current) return;
        const { lng, lat } = e.lngLat;
        const type = selectedTypeRef.current;
        try {
          const newPoint = await addPoint(plantId, {
            plant_id: plantId,
            point_type: type,
            latitude: lat,
            longitude: lng,
          });
          addMarker(maplibregl, map, newPoint);
          setPoints((prev) => [...prev, newPoint]);
          setAddMode(false);
          toast.success(`${POINT_LABELS[type]} noktası eklendi`);
        } catch (err) {
          toast.error(err instanceof Error ? err.message : "Nokta eklenemedi");
        }
      });

      map.on("mousemove", () => {
        map.getCanvas().style.cursor = addModeRef.current ? "crosshair" : "";
      });
    });

    return () => {
      markersRef.current.forEach((m) => m.remove());
      markersRef.current = [];
      if (resizeObsRef.current) {
        resizeObsRef.current.disconnect();
        resizeObsRef.current = null;
      }
      if (mapInstance.current) {
        mapInstance.current.remove();
        mapInstance.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const pointTypes: PointType[] = ["weather", "turbine", "panel_array"];

  return (
    <div className="relative h-[calc(100vh-12rem)] min-h-96 rounded-xl overflow-hidden border">
      {/* MapLibre kendi CSS'inde .maplibregl-map'e position:relative atar ve Tailwind
          `absolute inset-0`'ı ezer → div 0px'e çöker. h-full ile gerçek yükseklik ver. */}
      <div ref={mapRef} className="h-full w-full" />

      {/* Floating panel */}
      <Card className="absolute right-4 top-4 z-10 w-64 shadow-lg">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Nokta Ekle</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <div className="flex flex-col gap-1">
            {pointTypes.map((type) => {
              const Icon = POINT_ICONS[type];
              return (
                <button
                  key={type}
                  onClick={() => setSelectedType(type)}
                  className={cn(
                    "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors",
                    selectedType === type
                      ? "bg-muted font-medium"
                      : "hover:bg-muted/50 text-muted-foreground"
                  )}
                >
                  <div
                    className="size-3 rounded-full"
                    style={{ backgroundColor: POINT_COLORS[type] }}
                  />
                  <Icon className="size-4" />
                  + {POINT_LABELS[type]}
                </button>
              );
            })}
          </div>

          <Button
            size="sm"
            variant={addMode ? "destructive" : "default"}
            onClick={() => setAddMode((v) => !v)}
          >
            {addMode ? "İptal" : "Haritaya Tıkla"}
          </Button>

          {addMode && (
            <p className="text-xs text-muted-foreground text-center">
              {POINT_LABELS[selectedType]} eklemek için haritaya tıkla
            </p>
          )}

          {points.length > 0 ? (
            <div className="flex flex-col gap-1 border-t pt-2">
              <p className="text-xs font-medium text-muted-foreground mb-1">
                Nokta listesi ({points.length})
              </p>
              {points.map((pt, i) => (
                <div key={pt.point_id ?? i} className="flex items-center gap-2">
                  <div
                    className="size-2 shrink-0 rounded-full"
                    style={{ backgroundColor: POINT_COLORS[pt.point_type] }}
                  />
                  <span className="text-xs truncate">
                    {POINT_LABELS[pt.point_type]} ({pt.latitude.toFixed(3)},{" "}
                    {pt.longitude.toFixed(3)})
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground text-center border-t pt-2">
              Henüz nokta yok
            </p>
          )}
        </CardContent>
      </Card>

      {!mapLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-muted/50">
          <Skeleton className="h-8 w-32" />
        </div>
      )}
    </div>
  );
}
