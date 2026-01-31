"use client";

import Link from "next/link";
import { useMutation, useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { Button, getButtonClassName } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { getJobResult, getLibrary, deleteJob } from "@/lib/api";
import type { LibraryItem } from "@/lib/types";
import { formatDateTime, formatJobTitle } from "@/lib/format";

function StatusBadge({ status }: { status: LibraryItem["status"] }) {
  if (status === "DONE") {
    return <Badge variant="success">Terminé</Badge>;
  }
  return <Badge variant="danger">Échec</Badge>;
}

export default function LibraryPage() {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["library"],
    queryFn: getLibrary,
  });

  const deleteMutation = useMutation({
    mutationFn: (jobId: string) => deleteJob(jobId),
    onSuccess: () => {
      toast.success("Transcription supprimée.");
      refetch();
    },
    onError: (err) => {
      toast.error(err instanceof Error ? err.message : "Suppression impossible.");
    },
  });

  const handleDownload = async (jobId: string) => {
    try {
      const result = await getJobResult(jobId);
      const url =
        result.pdfUrl || result.musicXmlUrl || result.tabTxtUrl || result.tabJsonUrl || result.midiUrl;
      if (!url) {
        toast.error("Aucun fichier disponible pour ce job.");
        return;
      }
      window.open(url, "_blank", "noopener,noreferrer");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Téléchargement impossible.");
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-100">Bibliothèque</h1>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Retrouvez vos transcriptions et accédez aux téléchargements.
          </p>
        </div>
        <Link href="/new" className={getButtonClassName("primary")}>
          Nouvelle transcription
        </Link>
      </div>

      {isLoading && <Card>Chargement de la bibliothèque...</Card>}
      {error && (
        <Card className="space-y-3">
          <p className="text-sm text-red-600 dark:text-red-300">
            Impossible de charger la bibliothèque.
          </p>
          <Button variant="secondary" onClick={() => refetch()}>
            Réessayer
          </Button>
        </Card>
      )}

      {!isLoading && !error && (data?.items?.length ?? 0) === 0 && (
        <Card>Aucune transcription pour le moment.</Card>
      )}

      <div className="grid gap-4">
        {data?.items?.map((item) => (
          <Card key={item.jobId} className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
                  {formatJobTitle(item)}
                </h2>
                <StatusBadge status={item.status} />
                {item.confidence != null && (
                  <Badge variant="neutral">Confiance {Math.round(item.confidence * 100)}%</Badge>
                )}
              </div>
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                {formatDateTime(item.createdAt)}
              </p>
              {item.outputType && (
                <p className="text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                  Sortie : {item.outputType}
                </p>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              <Link href={`/jobs/${item.jobId}`} className={getButtonClassName("secondary")}>
                Ouvrir
              </Link>
              <Button variant="ghost" onClick={() => handleDownload(item.jobId)}>
                Télécharger
              </Button>
              <Button
                variant="danger"
                onClick={() => deleteMutation.mutate(item.jobId)}
                disabled={deleteMutation.isPending}
              >
                Supprimer
              </Button>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
