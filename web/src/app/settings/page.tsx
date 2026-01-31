"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import {
  getApiUrlOverride,
  getPreferredExportFormat,
  setApiUrlOverride,
  setPreferredExportFormat,
  type ExportFormat,
} from "@/lib/storage";

export default function SettingsPage() {
  const [apiUrl, setApiUrl] = useState("");
  const [format, setFormat] = useState<ExportFormat>("pdf");

  useEffect(() => {
    setApiUrl(getApiUrlOverride() ?? "");
    setFormat(getPreferredExportFormat() ?? "pdf");
  }, []);

  const handleSave = () => {
    setApiUrlOverride(apiUrl.trim());
    setPreferredExportFormat(format);
    toast.success("Paramètres enregistrés.");
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-100">Paramètres</h1>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Configurez votre backend et votre format d’export préféré.
        </p>
      </div>

      <Card className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="apiUrl">URL backend (optionnel)</Label>
          <Input
            id="apiUrl"
            placeholder="http://localhost:8000"
            value={apiUrl}
            onChange={(event) => setApiUrl(event.target.value)}
          />
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            Laissez vide pour utiliser `NEXT_PUBLIC_API_URL`.
          </p>
        </div>
        <div className="space-y-2">
          <Label htmlFor="format">Format d’export préféré</Label>
          <Select id="format" value={format} onChange={(event) => setFormat(event.target.value as ExportFormat)}>
            <option value="pdf">PDF</option>
            <option value="musicxml">MusicXML</option>
            <option value="tab">TAB texte</option>
            <option value="midi">MIDI</option>
          </Select>
        </div>
        <Button onClick={handleSave}>Enregistrer</Button>
      </Card>
    </div>
  );
}
