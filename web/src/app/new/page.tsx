"use client";

import { useRouter } from "next/navigation";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { toast } from "sonner";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { createJob } from "@/lib/api";

const youtubeRegex =
  /^(https?:\/\/)?(www\.)?(youtube\.com\/(watch\?v=|shorts\/|embed\/)|youtu\.be\/)[\w-]{6,}/i;

const formSchema = z
  .object({
    audioFile: z.instanceof(File).optional().nullable(),
    youtubeUrl: z.string().trim().optional().or(z.literal("")),
    outputType: z.enum(["tab", "score", "both"]),
    tuning: z.string().trim().min(1, "Tuning requis."),
    capo: z.number().min(0).max(12),
    quality: z.enum(["fast", "accurate"]),
    startSeconds: z.number().int().min(0).max(12 * 60).optional(),
    endSeconds: z.number().int().min(0).max(12 * 60).optional(),
  })
  .superRefine((data, ctx) => {
    const hasFile = Boolean(data.audioFile);
    const hasUrl = Boolean(data.youtubeUrl);
    if (!hasFile && !hasUrl) {
      ctx.addIssue({
        code: "custom",
        message: "Ajoutez un fichier audio ou une URL YouTube.",
        path: ["audioFile"],
      });
    }
    if (hasFile && hasUrl) {
      ctx.addIssue({
        code: "custom",
        message: "Choisissez soit un fichier audio, soit une URL YouTube.",
        path: ["audioFile"],
      });
    }
    if (hasUrl && data.youtubeUrl && !youtubeRegex.test(data.youtubeUrl)) {
      ctx.addIssue({
        code: "custom",
        message: "URL YouTube invalide.",
        path: ["youtubeUrl"],
      });
    }
    if (data.startSeconds != null && data.endSeconds != null && data.endSeconds <= data.startSeconds) {
      ctx.addIssue({
        code: "custom",
        message: "Le temps de fin doit être supérieur au temps de début.",
        path: ["endSeconds"],
      });
    }
  });

type FormValues = z.infer<typeof formSchema>;

export default function NewTranscriptionPage() {
  const router = useRouter();
  const {
    register,
    handleSubmit,
    setValue,
    formState: { errors, isSubmitting },
  } = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      outputType: "both",
      tuning: "EADGBE",
      capo: 0,
      quality: "fast",
      youtubeUrl: "",
      audioFile: null,
      startSeconds: undefined,
      endSeconds: undefined,
    },
  });

  const onSubmit = async (values: FormValues) => {
    try {
      const response = await createJob({
        outputType: values.outputType,
        tuning: values.tuning,
        capo: values.capo,
        quality: values.quality,
        audioFile: values.audioFile,
        youtubeUrl: values.youtubeUrl || null,
        startSeconds: values.startSeconds,
        endSeconds: values.endSeconds,
      });
      toast.success("Job lancé.");
      router.push(`/jobs/${response.jobId}`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Impossible de lancer la transcription.");
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-100">Nouvelle transcription</h1>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Importez un fichier audio ou collez une URL YouTube.
        </p>
      </div>

      <Card>
        <form className="space-y-5" onSubmit={handleSubmit(onSubmit)}>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="audioFile">Fichier audio</Label>
              <Input
                id="audioFile"
                type="file"
                accept=".mp3,.wav,.m4a,.aac"
                onChange={(event) => {
                  const file = event.target.files?.[0] ?? null;
                  setValue("audioFile", file, { shouldValidate: true });
                }}
              />
              {errors.audioFile && (
                <p className="text-xs text-red-600 dark:text-red-300">{errors.audioFile.message}</p>
              )}
            </div>
            <div className="space-y-2">
              <Label htmlFor="youtubeUrl">URL YouTube</Label>
              <Input id="youtubeUrl" placeholder="https://youtu.be/..." {...register("youtubeUrl")} />
              {errors.youtubeUrl && (
                <p className="text-xs text-red-600 dark:text-red-300">{errors.youtubeUrl.message}</p>
              )}
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-4">
            <div className="space-y-2">
              <Label htmlFor="outputType">Sortie</Label>
              <Select id="outputType" {...register("outputType")}>
                <option value="tab">Tablature</option>
                <option value="score">Partition</option>
                <option value="both">Les deux</option>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="tuning">Tuning</Label>
              <Input id="tuning" placeholder="EADGBE" {...register("tuning")} />
              {errors.tuning && <p className="text-xs text-red-600 dark:text-red-300">{errors.tuning.message}</p>}
            </div>
            <div className="space-y-2">
              <Label htmlFor="capo">Capo</Label>
              <Input
                id="capo"
                type="number"
                min={0}
                max={12}
                {...register("capo", { valueAsNumber: true })}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="quality">Qualité</Label>
              <Select id="quality" {...register("quality")}>
                <option value="fast">Rapide</option>
                <option value="accurate">Précis</option>
              </Select>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="startSeconds">Début (secondes)</Label>
              <Input
                id="startSeconds"
                type="number"
                min={0}
                max={12 * 60}
                placeholder="0"
                {...register("startSeconds", {
                  setValueAs: (value) => (value === "" ? undefined : Number(value)),
                })}
              />
              {errors.startSeconds && (
                <p className="text-xs text-red-600 dark:text-red-300">{errors.startSeconds.message}</p>
              )}
            </div>
            <div className="space-y-2">
              <Label htmlFor="endSeconds">Fin (secondes)</Label>
              <Input
                id="endSeconds"
                type="number"
                min={0}
                max={12 * 60}
                placeholder="120"
                {...register("endSeconds", {
                  setValueAs: (value) => (value === "" ? undefined : Number(value)),
                })}
              />
              {errors.endSeconds && (
                <p className="text-xs text-red-600 dark:text-red-300">{errors.endSeconds.message}</p>
              )}
            </div>
          </div>

          <div className="flex flex-wrap gap-3">
            <Button type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Transcription en cours..." : "Transcrire"}
            </Button>
            <Button type="button" variant="secondary" onClick={() => router.push("/library")}>
              Retour bibliothèque
            </Button>
          </div>
        </form>
      </Card>
    </div>
  );
}
