// dashboard/src/components/PipelinePanel.tsx
import React from "react";
import {
    Layers,
    Recycle,
    Gauge,
    HardDrive,
    ArrowRightLeft,
    Cpu,
    Boxes,
} from "lucide-react";

export interface PipelineMetrics {
    prefillTokens: number;
    prefixSkipTokens: number;
    adaptiveK: number;
    evictions: number;
    pageFaults: number;
    swapBytesOut: number;
    swapBytesIn: number;
    attentionBackend: string;
    /** Executor fuses decode steps into one model forward. */
    batchedDecode?: boolean;
    /** Fused decode forwards executed. */
    decodeBatches?: number;
    /** Sequences covered by those forwards. */
    decodeBatchSequences?: number;
    /** Largest fused forward so far. */
    maxDecodeBatch?: number;
}

interface PipelinePanelProps {
    metrics: PipelineMetrics;
}

function formatBytes(bytes: number): string {
    if (!Number.isFinite(bytes) || bytes <= 0) return "0 MB";
    const mb = bytes / (1024 * 1024);
    if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
    if (mb >= 1) return `${mb.toFixed(1)} MB`;
    return `${Math.round(bytes / 1024)} KB`;
}

function attentionLabel(backend: string): string {
    if (backend === "custom-cuda-f16") return "CUDA f16";
    if (backend === "gather") return "Gather";
    if (backend === "mock") return "Mock";
    return "Unknown";
}

/** Mean sequences per fused decode forward, e.g. "7.8". */
function formatBatchSize(batches?: number, sequences?: number): string {
    if (!batches || batches <= 0) return "0";
    return ((sequences || 0) / batches).toFixed(1);
}

function NumberTile({
    label,
    value,
    detail,
    icon: Icon,
    accent,
}: {
    label: string;
    value: string;
    detail: string;
    icon: React.ComponentType<{ className?: string }>;
    accent: string;
}) {
    return (
        <div className="bg-slate-900/90 border border-slate-800 p-4 rounded-2xl shadow-lg min-w-0">
            <div className="flex items-center justify-between text-slate-400 mb-2">
                <span className="text-[11px] font-semibold uppercase tracking-wider">{label}</span>
                <Icon className={`w-4 h-4 shrink-0 ${accent}`} />
            </div>
            <div className="text-xl font-black text-white tracking-tight truncate">{value}</div>
            <div className={`text-[10px] font-mono mt-0.5 truncate ${accent}`}>{detail}</div>
        </div>
    );
}

export const PipelinePanel: React.FC<PipelinePanelProps> = ({ metrics }) => {
    const prefill = Math.max(0, metrics.prefillTokens || 0);
    const skipped = Math.max(0, metrics.prefixSkipTokens || 0);
    const forwarded = prefill + skipped;
    const reusePercent = forwarded > 0 ? (skipped / forwarded) * 100 : 0;
    const swapBytes = Math.max(0, metrics.swapBytesOut || 0) + Math.max(0, metrics.swapBytesIn || 0);
    const backend = metrics.attentionBackend || "unknown";

    return (
        <section className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 md:p-5 shadow-xl">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
                <div>
                    <h2 className="text-sm font-semibold tracking-wide text-white flex items-center gap-2">
                        <Layers className="w-4 h-4 text-emerald-400" />
                        KV pipeline · measured
                    </h2>
                    <p className="text-xs text-slate-400 mt-0.5">
                        Prefix hits skip forwarded tokens; tier counters are real bytes moved.
                    </p>
                </div>
                <span className="self-start sm:self-auto px-2.5 py-1 rounded-full border border-slate-700 bg-slate-950 text-[11px] font-mono text-slate-300">
                    attention: {backend}
                </span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3.5">
                <NumberTile
                    label="Prefill compute"
                    value={prefill.toLocaleString()}
                    detail="tokens forwarded"
                    icon={Cpu}
                    accent="text-cyan-400"
                />
                <NumberTile
                    label="Prefix reuse"
                    value={`${reusePercent.toFixed(0)}%`}
                    detail={`${skipped.toLocaleString()} tokens skipped`}
                    icon={Recycle}
                    accent="text-emerald-400"
                />
                <NumberTile
                    label="Adaptive K"
                    value={String(metrics.adaptiveK || 0)}
                    detail="UCB1 draft arm"
                    icon={Gauge}
                    accent="text-amber-400"
                />
                <NumberTile
                    label="Evictions"
                    value={(metrics.evictions || 0).toLocaleString()}
                    detail={`${(metrics.pageFaults || 0).toLocaleString()} page faults`}
                    icon={HardDrive}
                    accent="text-rose-400"
                />
                <NumberTile
                    label="Tier swap"
                    value={formatBytes(swapBytes)}
                    detail={`out ${formatBytes(metrics.swapBytesOut)} · in ${formatBytes(metrics.swapBytesIn)}`}
                    icon={ArrowRightLeft}
                    accent="text-violet-400"
                />
                <NumberTile
                    label="Decode batch"
                    value={
                        metrics.batchedDecode
                            ? `${formatBatchSize(metrics.decodeBatches, metrics.decodeBatchSequences)}×`
                            : "1×"
                    }
                    detail={
                        metrics.batchedDecode
                            ? `${(metrics.decodeBatches || 0).toLocaleString()} fused · max ${metrics.maxDecodeBatch || 0}`
                            : "not fused by executor"
                    }
                    icon={Boxes}
                    accent="text-teal-400"
                />
                <NumberTile
                    label="Attention"
                    value={attentionLabel(backend)}
                    detail={backend}
                    icon={Layers}
                    accent="text-blue-400"
                />
            </div>
        </section>
    );
};
