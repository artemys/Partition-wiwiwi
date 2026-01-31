"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useMutation, useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { deleteJob, getJobResult, getJobStatus } from "@/lib/api";
import { formatDateTime } from "@/lib/format";
import { getPreferredExportFormat } from "@/lib/storage";

export default function JobDetailsPage() {
  const params = useParams();
  const router = useRouter();
  const jobId = Array.isArray(params.jobId) ? params.jobId[0] : params.jobId;
  const resolvedJobId = typeof jobId === "string" ? jobId : "";
  const [tabPreview, setTabPreview] = useState<string | null>(null);

  const {
    data: status,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["job", resolvedJobId],
    queryFn: () => getJobStatus(resolvedJobId),
    enabled: Boolean(resolvedJobId),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return 2000;
      return data.status === "DONE" || data.status === "FAILED" ? false : 2000;
    },
  });

  const { data: result } = useQuery({
    queryKey: ["job-result", resolvedJobId],
    queryFn: () => getJobResult(resolvedJobId),
    enabled: status?.status === "DONE" && Boolean(resolvedJobId),
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteJob(resolvedJobId),
    onSuccess: () => {
      toast.success("Transcription supprimée.");
      router.push("/library");
    },
    onError: (err) => {
      toast.error(err instanceof Error ? err.message : "Suppression impossible.");
    },
  });

  const downloadUrl = useMemo(() => {
    if (!result) return null;
    const preference = getPreferredExportFormat();
    if (preference === "pdf" && result.pdfUrl) return result.pdfUrl;
    if (preference === "musicxml" && result.musicXmlUrl) return result.musicXmlUrl;
    if (preference === "tab" && result.tabTxtUrl) return result.tabTxtUrl;
    if (preference === "midi" && result.midiUrl) return result.midiUrl;
    return result.pdfUrl || result.musicXmlUrl || result.tabTxtUrl || result.tabJsonUrl || result.midiUrl || null;
  }, [result]);

  useEffect(() => {
    const tabUrl = result?.tabTxtUrl;
    if (!tabUrl) {
      setTabPreview(null);
      return;
    }
    let cancelled = false;
    fetch(tabUrl)
      .then((res) => res.text())
      .then((text) => {
        if (!cancelled) {
          setTabPreview(text);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setTabPreview("Impossible de charger la tablature.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [result?.tabTxtUrl]);

  if (!resolvedJobId) {
    return <Card>Job introuvable.</Card>;
  }

  if (isLoading) {
    return <Card>Chargement du job...</Card>;
  }

  if (error || !status) {
    return <Card>Impossible de charger le job.</Card>;
  }

  const statusBadge =
    status.status === "DONE"
      ? "success"
      : status.status === "FAILED"
        ? "danger"
        : "warning";

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-100">Détails du job</h1>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">ID: {resolvedJobId}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" onClick={() => router.push("/library")}>
            Retour bibliothèque
          </Button>
          <Button variant="danger" onClick={() => deleteMutation.mutate()} disabled={deleteMutation.isPending}>
            Supprimer
          </Button>
        </div>
      </div>

      <Card className="space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <Badge variant={statusBadge as "success" | "danger" | "warning"}>
            {status.status === "DONE"
              ? "Terminé"
              : status.status === "FAILED"
                ? "Échec"
                : status.status === "RUNNING"
                  ? "En cours"
                  : "En attente"}
          </Badge>
          {status.confidence != null && (
            <Badge variant="neutral">Confiance {Math.round(status.confidence * 100)}%</Badge>
          )}
          {status.createdAt && (
            <span className="text-xs text-zinc-500 dark:text-zinc-400">
              {formatDateTime(status.createdAt)}
            </span>
          )}
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm text-zinc-600 dark:text-zinc-300">
            <span>Étape : {status.stage ?? "—"}</span>
            <span>{status.progress}%</span>
          </div>
          <Progress value={status.progress} />
        </div>

        {status.status === "FAILED" && status.errorMessage && (
          <div className="rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-200">
            {status.errorMessage}
          </div>
        )}

        {status.status === "DONE" && (
          <div className="flex flex-wrap gap-2">
            {downloadUrl && (
              <Button onClick={() => window.open(downloadUrl, "_blank", "noopener,noreferrer")}>Télécharger</Button>
            )}
            {result?.tabTxtUrl && (
              <Button
                variant="secondary"
                onClick={() =>
                  window.open(result.tabTxtUrl ?? undefined, "_blank", "noopener,noreferrer")
                }
              >
                Ouvrir TAB
              </Button>
            )}
            {result?.pdfUrl && (
              <Button
                variant="secondary"
                onClick={() => window.open(result.pdfUrl ?? undefined, "_blank", "noopener,noreferrer")}
              >
                Ouvrir PDF
              </Button>
            )}
          </div>
        )}
      </Card>

      {status.status === "DONE" && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card className="space-y-3">
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Prévisualisation TAB</h2>
            {result?.tabTxtUrl ? (
              <pre className="max-h-[520px] overflow-auto rounded-xl bg-zinc-900 p-4 text-xs text-zinc-100">
                {tabPreview ?? "Chargement du TAB..."}
              </pre>
            ) : (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">Aucune tablature disponible.</p>
            )}
          </Card>
          <Card className="space-y-3">
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Prévisualisation PDF</h2>
            {result?.pdfUrl ? (
              <iframe
                src={result.pdfUrl}
                className="h-[520px] w-full rounded-xl border border-zinc-200 dark:border-zinc-800"
                title="Prévisualisation PDF"
              />
            ) : (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">Aucun PDF disponible.</p>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}
