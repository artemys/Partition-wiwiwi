"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { toast } from "sonner";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import {
  getPlayabilitySpan,
  getPreferLowFrets,
  type PlayabilitySpan,
} from "@/lib/storage";
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
    transcriptionMode: z.enum(["best_free", "monophonic_tuner", "polyphonic_basic_pitch"]),
    arrangement: z.enum(["lead", "poly"]),
    confidenceThreshold: z.number().min(0.05).max(1),
    onsetWindowMs: z.number().int().min(10).max(250),
    maxJumpSemitones: z.number().int().min(1).max(24),
    gridResolution: z.enum(["auto", "eighth", "sixteenth"]),
    onsetDetection: z.boolean(),
    onsetAlign: z.boolean(),
    onsetGate: z.boolean(),
    minNoteDurationMs: z.number().int().min(10).max(500),
    midiMin: z.number().int().min(0).max(127),
    midiMax: z.number().int().min(0).max(127),
    snapToleranceMs: z.number().int().min(0).max(250),
    polyMaxNotes: z.number().int().min(1).max(8),
    tabMaxFret: z.number().int().min(0).max(24),
    tabMaxFretSpanChord: z.number().int().min(1).max(12),
    tabMaxPositionJump: z.number().int().min(0).max(12),
    tabMaxNotesPerChord: z.number().int().min(1).max(6),
    tabMeasuresPerSystem: z.number().int().min(1).max(8),
    tabWrapColumns: z.number().int().min(40).max(180),
    tabTokenWidth: z.number().int().min(2).max(5),
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
      transcriptionMode: "best_free",
      arrangement: "lead",
      confidenceThreshold: 0.2,
      onsetWindowMs: 60,
      maxJumpSemitones: 7,
      gridResolution: "auto",
      onsetDetection: true,
      onsetAlign: true,
      onsetGate: false,
      minNoteDurationMs: 60,
      midiMin: 40,
      midiMax: 88,
      snapToleranceMs: 35,
      polyMaxNotes: 3,
      tabMaxFret: 15,
      tabMaxFretSpanChord: 5,
      tabMaxPositionJump: 4,
      tabMaxNotesPerChord: 3,
      tabMeasuresPerSystem: 2,
      tabWrapColumns: 80,
      tabTokenWidth: 3,
      youtubeUrl: "",
      audioFile: null,
      startSeconds: undefined,
      endSeconds: undefined,
    },
  });

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [arrangementHint, setArrangementHint] = useState<FormValues["arrangement"]>("lead");
  const [playabilitySpan] = useState<PlayabilitySpan>(() => getPlayabilitySpan());
  const [preferLowFrets] = useState(() => getPreferLowFrets());

  const onSubmit = async (values: FormValues) => {
    try {
      const response = await createJob({
        outputType: values.outputType,
        tuning: values.tuning,
        capo: values.capo,
        quality: values.quality,
        transcriptionMode: values.transcriptionMode,
        arrangement: values.arrangement,
        confidenceThreshold: values.confidenceThreshold,
        onsetWindowMs: values.onsetWindowMs,
        maxJumpSemitones: values.maxJumpSemitones,
        gridResolution: values.gridResolution,
        onsetDetection: values.onsetDetection,
        onsetAlign: values.onsetAlign,
        onsetGate: values.onsetGate,
        minNoteDurationMs: values.minNoteDurationMs,
        midiMin: values.midiMin,
        midiMax: values.midiMax,
        snapToleranceMs: values.snapToleranceMs,
        polyMaxNotes: values.polyMaxNotes,
        tabMaxFret: values.tabMaxFret,
        tabMaxFretSpanChord: values.tabMaxFretSpanChord,
        tabMaxPositionJump: values.tabMaxPositionJump,
        tabMaxNotesPerChord: values.tabMaxNotesPerChord,
        tabMeasuresPerSystem: values.tabMeasuresPerSystem,
        tabWrapColumns: values.tabWrapColumns,
        tabTokenWidth: values.tabTokenWidth,
        handSpan: playabilitySpan,
        preferLowFrets,
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
          <div className="space-y-2">
            <Label htmlFor="transcriptionMode">Mode automatique</Label>
            <Select id="transcriptionMode" {...register("transcriptionMode")}>
              <option value="best_free">Recommandé — best_free (Demucs + Basic Pitch)</option>
              <option value="monophonic_tuner">Auto (type accordeur) — riffs/lead</option>
              <option value="polyphonic_basic_pitch">Auto (polyphonique) — accords</option>
            </Select>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              “best_free” isole mieux la guitare et vise une partition + tablature plus cohérentes.
            </p>
          </div>

          <Card className="space-y-4 border border-zinc-200 bg-white/50 dark:border-zinc-800 dark:bg-zinc-950/40">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <p className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">Réglages avancés</p>
                <p className="text-xs text-zinc-500 dark:text-zinc-400">
                  Pour améliorer la lisibilité rythmique et réduire les notes parasites.
                </p>
              </div>
              <Button type="button" variant="ghost" onClick={() => setShowAdvanced((v) => !v)}>
                {showAdvanced ? "Masquer" : "Afficher"}
              </Button>
            </div>

            {showAdvanced && (
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="arrangement">Arrangement</Label>
                  <Select
                    id="arrangement"
                    {...register("arrangement", {
                      onChange: (event) => setArrangementHint(event.target.value as FormValues["arrangement"]),
                    })}
                  >
                    <option value="lead">Lead (monodie)</option>
                    <option value="poly">Poly (accords)</option>
                  </Select>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">
                    Lead = 1 note max à un instant. Poly = conserve les accords.
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="gridResolution">Grille rythmique</Label>
                  <Select id="gridResolution" {...register("gridResolution")}>
                    <option value="auto">Auto (selon qualité)</option>
                    <option value="eighth">Croches (1/8)</option>
                    <option value="sixteenth">Doubles-croches (1/16)</option>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="confidenceThreshold">Seuil de confiance</Label>
                  <Input
                    id="confidenceThreshold"
                    type="number"
                    step="0.01"
                    min={0.05}
                    max={1}
                    {...register("confidenceThreshold", { valueAsNumber: true })}
                  />
                  {errors.confidenceThreshold && (
                    <p className="text-xs text-red-600 dark:text-red-300">
                      {errors.confidenceThreshold.message}
                    </p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="minNoteDurationMs">Durée min note (ms)</Label>
                  <Input
                    id="minNoteDurationMs"
                    type="number"
                    min={10}
                    max={500}
                    step={1}
                    {...register("minNoteDurationMs", { valueAsNumber: true })}
                  />
                  {errors.minNoteDurationMs && (
                    <p className="text-xs text-red-600 dark:text-red-300">{errors.minNoteDurationMs.message}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="onsetWindowMs">Fenêtre onset (ms)</Label>
                  <Input
                    id="onsetWindowMs"
                    type="number"
                    min={10}
                    max={250}
                    step={1}
                    {...register("onsetWindowMs", { valueAsNumber: true })}
                  />
                  {errors.onsetWindowMs && (
                    <p className="text-xs text-red-600 dark:text-red-300">{errors.onsetWindowMs.message}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="snapToleranceMs">Tolérance snap (ms)</Label>
                  <Input
                    id="snapToleranceMs"
                    type="number"
                    min={0}
                    max={250}
                    step={1}
                    {...register("snapToleranceMs", { valueAsNumber: true })}
                  />
                  {errors.snapToleranceMs && (
                    <p className="text-xs text-red-600 dark:text-red-300">{errors.snapToleranceMs.message}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="maxJumpSemitones">Saut max (demi-tons)</Label>
                  <Input
                    id="maxJumpSemitones"
                    type="number"
                    min={1}
                    max={24}
                    step={1}
                    {...register("maxJumpSemitones", { valueAsNumber: true })}
                  />
                  {errors.maxJumpSemitones && (
                    <p className="text-xs text-red-600 dark:text-red-300">
                      {errors.maxJumpSemitones.message}
                    </p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="polyMaxNotes">Poly: max notes simultanées</Label>
                  <Input
                    id="polyMaxNotes"
                    type="number"
                    min={1}
                    max={8}
                    step={1}
                    {...register("polyMaxNotes", { valueAsNumber: true })}
                  />
                  {errors.polyMaxNotes && (
                    <p className="text-xs text-red-600 dark:text-red-300">{errors.polyMaxNotes.message}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label>Onsets (best_free)</Label>
                  <div className="space-y-2 rounded-xl border border-zinc-200 bg-white/60 p-3 dark:border-zinc-800 dark:bg-zinc-950/60">
                    <Checkbox {...register("onsetDetection")}>Détection d’onsets</Checkbox>
                    <Checkbox {...register("onsetAlign")}>Aligner start sur onsets (snap)</Checkbox>
                    <Checkbox {...register("onsetGate")}>Gate strict (supprime hors onsets)</Checkbox>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="midiMin">Plage MIDI min/max</Label>
                  <div className="grid grid-cols-2 gap-3">
                    <Input
                      id="midiMin"
                      type="number"
                      min={0}
                      max={127}
                      step={1}
                      {...register("midiMin", { valueAsNumber: true })}
                    />
                    <Input
                      id="midiMax"
                      type="number"
                      min={0}
                      max={127}
                      step={1}
                      {...register("midiMax", { valueAsNumber: true })}
                    />
                  </div>
                  {(errors.midiMin || errors.midiMax) && (
                    <p className="text-xs text-red-600 dark:text-red-300">
                      {errors.midiMin?.message || errors.midiMax?.message}
                    </p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label>Contraintes de doigté (TAB)</Label>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <Label htmlFor="tabMaxFret" className="text-xs">
                        Frette max
                      </Label>
                      <Input
                        id="tabMaxFret"
                        type="number"
                        min={0}
                        max={24}
                        step={1}
                        {...register("tabMaxFret", { valueAsNumber: true })}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="tabMaxFretSpanChord" className="text-xs">
                        Écart max (accord)
                      </Label>
                      <Input
                        id="tabMaxFretSpanChord"
                        type="number"
                        min={1}
                        max={12}
                        step={1}
                        {...register("tabMaxFretSpanChord", { valueAsNumber: true })}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="tabMaxPositionJump" className="text-xs">
                        Saut position (frettes)
                      </Label>
                      <Input
                        id="tabMaxPositionJump"
                        type="number"
                        min={0}
                        max={12}
                        step={1}
                        {...register("tabMaxPositionJump", { valueAsNumber: true })}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="tabMaxNotesPerChord" className="text-xs">
                        Notes max/accord
                      </Label>
                      <Input
                        id="tabMaxNotesPerChord"
                        type="number"
                        min={1}
                        max={6}
                        step={1}
                        {...register("tabMaxNotesPerChord", { valueAsNumber: true })}
                      />
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Rendu TAB texte</Label>
                  <div className="grid grid-cols-3 gap-3">
                    <div className="space-y-1">
                      <Label htmlFor="tabWrapColumns" className="text-xs">
                        Wrap colonnes
                      </Label>
                      <Input
                        id="tabWrapColumns"
                        type="number"
                        min={40}
                        max={180}
                        step={1}
                        {...register("tabWrapColumns", { valueAsNumber: true })}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="tabTokenWidth" className="text-xs">
                        Largeur token
                      </Label>
                      <Input
                        id="tabTokenWidth"
                        type="number"
                        min={2}
                        max={5}
                        step={1}
                        {...register("tabTokenWidth", { valueAsNumber: true })}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="tabMeasuresPerSystem" className="text-xs">
                        Mesures/ligne
                      </Label>
                      <Input
                        id="tabMeasuresPerSystem"
                        type="number"
                        min={1}
                        max={8}
                        step={1}
                        {...register("tabMeasuresPerSystem", { valueAsNumber: true })}
                      />
                    </div>
                  </div>
                </div>

                <div className="rounded-xl border border-zinc-200 bg-white/60 p-3 text-xs text-zinc-600 dark:border-zinc-800 dark:bg-zinc-950/60 dark:text-zinc-300">
                  <p className="font-semibold text-zinc-800 dark:text-zinc-100">Astuce</p>
                  <p className="mt-1">
                    {arrangementHint === "lead"
                      ? "En lead, l’app force une mélodie plus “humaine” (1 note), idéale pour riffs."
                      : "En poly, l’app tente de conserver les accords — plus exigeant mais plus fidèle."}
                  </p>
                </div>
              </div>
            )}
          </Card>

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
