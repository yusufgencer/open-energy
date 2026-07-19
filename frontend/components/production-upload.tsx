"use client";

import { useState, useRef, useCallback } from "react";
import { UploadCloudIcon, FileTextIcon, CheckCircleIcon } from "lucide-react";
import { toast } from "sonner";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import { uploadProduction } from "@/lib/api";

interface Props {
  plantId: string;
}

export default function ProductionUpload({ plantId }: Props) {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<number | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.name.endsWith(".csv")) {
        toast.error("Yalnızca CSV dosyası kabul edilir");
        return;
      }
      setFileName(file.name);
      setLoading(true);
      setResult(null);
      try {
        const { rows } = await uploadProduction(plantId, file);
        setResult(rows);
        toast.success(`${rows} satır başarıyla yüklendi`);
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Yükleme hatası");
      } finally {
        setLoading(false);
      }
    },
    [plantId]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Üretim Verisi Yükle</CardTitle>
        <CardDescription>
          CSV formatı: <code className="font-mono text-xs">timestamp,power_mw</code>
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={cn(
            "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-10 cursor-pointer transition-colors",
            dragging
              ? "border-primary bg-primary/5"
              : "border-muted-foreground/25 hover:border-muted-foreground/50 hover:bg-muted/30"
          )}
        >
          <UploadCloudIcon className="size-8 text-muted-foreground" />
          <div className="text-center">
            <p className="text-sm font-medium">
              CSV dosyasını sürükle veya tıkla
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              .csv uzantılı dosya
            </p>
          </div>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          className="sr-only"
          onChange={onInputChange}
        />

        {loading && (
          <div className="flex flex-col gap-2">
            <p className="text-sm text-muted-foreground">Yükleniyor...</p>
            <Progress className="h-1" />
          </div>
        )}

        {result !== null && !loading && (
          <div className="flex items-center gap-2 rounded-lg bg-green-50 px-3 py-2 text-sm text-green-700">
            <CheckCircleIcon className="size-4" />
            <span>
              <strong>{fileName}</strong> — {result} satır yüklendi
            </span>
          </div>
        )}

        {fileName && !loading && result === null && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <FileTextIcon className="size-4" />
            {fileName}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
