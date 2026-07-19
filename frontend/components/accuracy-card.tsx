"use client";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { TriangleAlertIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import type { TrialResult } from "@/lib/types";

interface Props {
  result: TrialResult;
}

function skillBadgeVariant(skill: number): "default" | "secondary" | "destructive" {
  if (skill >= 0.3) return "default";   // green via primary
  if (skill >= 0.1) return "secondary"; // amber-ish via secondary
  return "destructive";
}

export default function AccuracyCard({ result }: Props) {
  const blocks: string[] = (() => {
    try { return JSON.parse(result.feature_blocks) as string[]; }
    catch { return []; }
  })();

  const nrmsePercent =
    result.test_nrmse !== null
      ? (result.test_nrmse * 100).toFixed(1)
      : null;

  const isNegativeSkill = result.skill_score < 0;

  return (
    <Card className={cn("flex flex-col gap-0", result.is_champion && "ring-2 ring-primary")}>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <div>
            <CardTitle className="text-sm">
              {result.horizon_hours}h horizon
            </CardTitle>
            <CardDescription className="font-mono text-xs mt-0.5">
              {result.model_family}
            </CardDescription>
          </div>
          <div className="flex flex-col items-end gap-1">
            {result.is_champion && (
              <Badge variant="default" className="text-xs">Şampiyon</Badge>
            )}
            <Badge variant={skillBadgeVariant(result.skill_score)} className="num text-xs">
              {result.skill_score >= 0 ? "▲" : "▼"} {result.skill_score.toFixed(3)}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {nrmsePercent !== null && (
          <div>
            <div className="flex justify-between text-xs text-muted-foreground mb-1">
              <span className="uppercase tracking-wide text-[10px]">nRMSE</span>
              <span className="num font-semibold text-foreground">
                {nrmsePercent}%
              </span>
            </div>
            <Progress
              value={Math.min(100, result.test_nrmse! * 100)}
              className="h-1.5"
            />
          </div>
        )}

        {isNegativeSkill && (
          <Alert variant="destructive" className="py-2">
            <TriangleAlertIcon className="size-3" />
            <AlertDescription className="text-xs">
              Bu model baseline&apos;ı yenmiyor
            </AlertDescription>
          </Alert>
        )}

        {blocks.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {blocks.map((b) => (
              <Badge key={b} variant="outline" className="text-xs font-mono">
                {b}
              </Badge>
            ))}
          </div>
        )}

        <div className="text-xs text-muted-foreground">
          NWP: <span className="font-mono">{result.nwp_source}</span>
        </div>

        {(result.point_policy || result.source_policy) && (
          <div className="flex flex-wrap items-center gap-1 border-t pt-2 text-[10px]">
            <span className="uppercase tracking-wide text-muted-foreground">
              Kazanan
            </span>
            {result.point_policy && (
              <Badge variant="secondary" className="font-mono text-[10px]">
                {result.point_policy}
              </Badge>
            )}
            {result.source_policy && (
              <>
                <span className="text-muted-foreground">×</span>
                <Badge variant="secondary" className="font-mono text-[10px]">
                  {result.source_policy}
                </Badge>
              </>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
