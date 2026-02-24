/**
 * DatasetPage — manage datasets, upload parquets, and trigger ingestion.
 *
 * Layout:
 *   Left panel  : dataset list + create form
 *   Right panel : selected dataset actions (upload, ingest, status)
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  createDataset,
  getIngestStatus,
  listDatasets,
  submitIngest,
  uploadParquets,
  type DatasetInfo,
} from '../api/platformClient';
import { useDatasetStore } from '../store/datasetStore';

export function DatasetPage() {
  const { activeDataset, setActiveDataset } = useDatasetStore();
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [createName, setCreateName] = useState('');
  const [createError, setCreateError] = useState('');
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploadStatus, setUploadStatus] = useState('');
  const [ingestJobName, setIngestJobName] = useState('');
  const [ingestState, setIngestState] = useState('');
  const [ingestPolling, setIngestPolling] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setDatasets(await listDatasets());
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Stop polling when job finishes
  useEffect(() => {
    if (!ingestPolling) return;
    const TERMINAL = ['ResourceReleased', 'RESOURCE_RELEASED', 'Succeeded', 'SUCCEEDED', 'FAILED', 'Failed'];
    if (TERMINAL.includes(ingestState)) {
      setIngestPolling(false);
      if (pollRef.current) clearInterval(pollRef.current);
    }
  }, [ingestState, ingestPolling]);

  const handleCreate = async () => {
    setCreateError('');
    if (!createName.trim()) return;
    try {
      await createDataset(createName.trim());
      setCreateName('');
      await refresh();
    } catch (e) {
      setCreateError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    const invalid = files.filter((f) => !f.name.toLowerCase().endsWith('.parquet'));
    if (invalid.length > 0) {
      setUploadStatus(`Only .parquet files allowed. Rejected: ${invalid.map((f) => f.name).join(', ')}`);
      return;
    }
    setUploadFiles(files);
    setUploadStatus('');
  };

  const handleUpload = async () => {
    if (!activeDataset || uploadFiles.length === 0) return;
    setUploadStatus('Uploading…');
    try {
      const result = await uploadParquets(activeDataset, uploadFiles);
      setUploadStatus(`Uploaded ${result.uploaded.length} file(s) to S3.`);
      setUploadFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = '';
      await refresh();
    } catch (e) {
      setUploadStatus(`Upload failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const handleIngest = async () => {
    if (!activeDataset) return;
    setIngestState('Submitting…');
    setIngestJobName('');
    try {
      const result = await submitIngest(activeDataset);
      setIngestJobName(result.job_name);
      setIngestState('Submitted');
      setIngestPolling(true);

      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const status = await getIngestStatus(activeDataset, result.job_name);
          setIngestState(status.state);
        } catch {
          // transient error — keep polling
        }
      }, 10_000);
    } catch (e) {
      setIngestState(`Failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const selectedInfo = datasets.find((d) => d.name === activeDataset);
  const stateColor =
    ingestState.includes('Fail') ? 'text-red-400'
    : ['ResourceReleased', 'Succeeded', 'SUCCEEDED', 'RESOURCE_RELEASED'].includes(ingestState) ? 'text-green-400'
    : ingestState ? 'text-yellow-400'
    : 'text-slate-400';

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── Left: dataset list ────────────────────────────────── */}
      <div className="flex w-64 shrink-0 flex-col border-r border-slate-700 bg-slate-900">
        <div className="border-b border-slate-700 p-3 text-sm font-semibold text-slate-300">
          Datasets
        </div>

        {/* Create dataset */}
        <div className="border-b border-slate-700 p-3">
          <div className="flex gap-2">
            <input
              className="flex-1 rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 placeholder-slate-500 outline-none focus:ring-1 focus:ring-blue-500"
              placeholder="new_dataset_name"
              value={createName}
              onChange={(e) => setCreateName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && void handleCreate()}
            />
            <button
              onClick={() => void handleCreate()}
              className="rounded bg-blue-600 px-2 py-1 text-xs text-white hover:bg-blue-500"
            >
              +
            </button>
          </div>
          {createError && <p className="mt-1 text-xs text-red-400">{createError}</p>}
        </div>

        {/* Dataset list */}
        <div className="flex-1 overflow-y-auto">
          {loading && (
            <p className="p-3 text-xs text-slate-400">Loading…</p>
          )}
          {!loading && datasets.length === 0 && (
            <p className="p-3 text-xs text-slate-500">No datasets yet.</p>
          )}
          {datasets.map((d) => (
            <button
              key={d.name}
              onClick={() => setActiveDataset(d.name)}
              className={`w-full border-b border-slate-800 px-3 py-2 text-left text-xs hover:bg-slate-800 ${
                activeDataset === d.name ? 'bg-slate-800 text-blue-300' : 'text-slate-300'
              }`}
            >
              <div className="font-medium">{d.name}</div>
              <div className="text-slate-500">
                {d.file_count} file{d.file_count !== 1 ? 's' : ''} ·{' '}
                {(d.size_bytes / 1024 / 1024).toFixed(1)} MB
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* ── Right: selected dataset actions ──────────────────── */}
      <div className="flex-1 overflow-y-auto p-6">
        {!activeDataset ? (
          <div className="flex h-full items-center justify-center">
            <p className="text-slate-500">Select or create a dataset to get started.</p>
          </div>
        ) : (
          <div className="mx-auto max-w-xl space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-slate-100">{activeDataset}</h2>
              {selectedInfo && (
                <span className="text-xs text-slate-400">
                  {selectedInfo.file_count} parquet file{selectedInfo.file_count !== 1 ? 's' : ''}
                </span>
              )}
            </div>

            {/* Upload */}
            <section className="rounded-lg border border-slate-700 bg-slate-900 p-4">
              <h3 className="mb-3 text-sm font-medium text-slate-300">Upload Parquet Files</h3>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".parquet"
                onChange={handleFileChange}
                className="block w-full text-xs text-slate-400 file:mr-3 file:rounded file:border-0 file:bg-slate-700 file:px-3 file:py-1 file:text-xs file:text-slate-200 hover:file:bg-slate-600"
              />
              {uploadFiles.length > 0 && (
                <p className="mt-2 text-xs text-slate-400">
                  {uploadFiles.length} file{uploadFiles.length !== 1 ? 's' : ''} selected
                </p>
              )}
              <button
                onClick={() => void handleUpload()}
                disabled={uploadFiles.length === 0}
                className="mt-3 rounded bg-blue-600 px-4 py-1.5 text-sm text-white hover:bg-blue-500 disabled:opacity-40"
              >
                Upload to S3
              </button>
              {uploadStatus && (
                <p className={`mt-2 text-xs ${uploadStatus.startsWith('Upload failed') ? 'text-red-400' : 'text-green-400'}`}>
                  {uploadStatus}
                </p>
              )}
            </section>

            {/* Ingest */}
            <section className="rounded-lg border border-slate-700 bg-slate-900 p-4">
              <h3 className="mb-1 text-sm font-medium text-slate-300">Ingest to Iceberg</h3>
              <p className="mb-3 text-xs text-slate-500">
                Reads <code className="text-slate-400">raw/{activeDataset}/</code> from S3 and
                writes to <code className="text-slate-400">iceberg.raw.{activeDataset}</code>.
              </p>
              <button
                onClick={() => void handleIngest()}
                disabled={ingestPolling}
                className="rounded bg-emerald-600 px-4 py-1.5 text-sm text-white hover:bg-emerald-500 disabled:opacity-40"
              >
                {ingestPolling ? 'Running…' : 'Start Ingestion'}
              </button>
              {ingestJobName && (
                <p className="mt-2 text-xs text-slate-400">Job: {ingestJobName}</p>
              )}
              {ingestState && (
                <p className={`mt-1 text-xs font-medium ${stateColor}`}>
                  State: {ingestState}
                </p>
              )}
            </section>

            {/* Open DSL Builder */}
            <p className="text-xs text-slate-500">
              After ingestion, open the{' '}
              <span className="text-blue-400">DSL Builder</span> tab to build and export
              your feature pipeline for this dataset.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
