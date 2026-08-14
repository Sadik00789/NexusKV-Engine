// dashboard/src/components/MetricsPanel.tsx
import React from "react";
import { Zap, Cpu, Activity, ShieldCheck, Timer, BookmarkCheck } from "lucide-react";

interface MetricsPanelProps {
    tokensPerSec: number;
    ttftMs: number;
    itlMs: number;
    allocatedBlocks: number;
    totalBlocks: number;
    memoryRatio: number;
    prefixCacheHitRatio: number;
    acceptanceRate: number;
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({
    tokensPerSec,
    ttftMs,
    itlMs,
    allocatedBlocks,
    totalBlocks,
    memoryRatio,
    prefixCacheHitRatio,
    acceptanceRate,
}) => {
    return (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3.5">
            {/* Throughput */}
            <div className="bg-slate-900/90 border border-slate-800 p-4 rounded-2xl flex flex-col justify-between hover:border-emerald-500/50 transition-colors shadow-lg">
                <div className="flex items-center justify-between text-slate-400 mb-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wider">Throughput</span>
                    <Zap className="w-4 h-4 text-emerald-400" />
                </div>
                <div>
                    <div className="text-2xl font-black text-white tracking-tight">
                        {tokensPerSec.toFixed(1)}
                        <span className="text-xs font-normal text-slate-400 ml-1">tok/s</span>
                    </div>
                    <div className="text-[10px] text-emerald-400 font-mono mt-0.5">Continuous Decodes</div>
                </div>
            </div>

            {/* Time to First Token (TTFT) */}
            <div className="bg-slate-900/90 border border-slate-800 p-4 rounded-2xl flex flex-col justify-between hover:border-cyan-500/50 transition-colors shadow-lg">
                <div className="flex items-center justify-between text-slate-400 mb-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wider">TTFT Latency</span>
                    <Timer className="w-4 h-4 text-cyan-400" />
                </div>
                <div>
                    <div className="text-2xl font-black text-white tracking-tight">
                        {ttftMs > 0 ? ttftMs.toFixed(1) : "—"}
                        <span className="text-xs font-normal text-slate-400 ml-1">ms</span>
                    </div>
                    <div className="text-[10px] text-cyan-400 font-mono mt-0.5">Time to 1st Token</div>
                </div>
            </div>

            {/* Inter-Token Latency (ITL) */}
            <div className="bg-slate-900/90 border border-slate-800 p-4 rounded-2xl flex flex-col justify-between hover:border-blue-500/50 transition-colors shadow-lg">
                <div className="flex items-center justify-between text-slate-400 mb-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wider">ITL Latency</span>
                    <Activity className="w-4 h-4 text-blue-400" />
                </div>
                <div>
                    <div className="text-2xl font-black text-white tracking-tight">
                        {itlMs > 0 ? itlMs.toFixed(1) : "—"}
                        <span className="text-xs font-normal text-slate-400 ml-1">ms</span>
                    </div>
                    <div className="text-[10px] text-blue-400 font-mono mt-0.5">Inter-Token Gap</div>
                </div>
            </div>

            {/* Prefix Cache Hit Ratio */}
            <div className="bg-slate-900/90 border border-slate-800 p-4 rounded-2xl flex flex-col justify-between hover:border-indigo-500/50 transition-colors shadow-lg">
                <div className="flex items-center justify-between text-slate-400 mb-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wider">Prefix Hit Rate</span>
                    <BookmarkCheck className="w-4 h-4 text-indigo-400" />
                </div>
                <div>
                    <div className="text-2xl font-black text-white tracking-tight">
                        {(prefixCacheHitRatio * 100).toFixed(0)}
                        <span className="text-xs font-normal text-slate-400 ml-1">%</span>
                    </div>
                    <div className="text-[10px] text-indigo-400 font-mono mt-0.5">Radix Trie Reuse</div>
                </div>
            </div>

            {/* VRAM Memory Utilization */}
            <div className="bg-slate-900/90 border border-slate-800 p-4 rounded-2xl flex flex-col justify-between hover:border-purple-500/50 transition-colors shadow-lg">
                <div className="flex items-center justify-between text-slate-400 mb-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wider">VRAM Pool</span>
                    <Cpu className="w-4 h-4 text-purple-400" />
                </div>
                <div>
                    <div className="text-2xl font-black text-white tracking-tight">
                        {(memoryRatio * 100).toFixed(1)}
                        <span className="text-xs font-normal text-slate-400 ml-1">%</span>
                    </div>
                    <div className="text-[10px] text-purple-400 font-mono mt-0.5">
                        {allocatedBlocks} / {totalBlocks} Pages
                    </div>
                </div>
            </div>

            {/* Speculative Acceptance */}
            <div className="bg-slate-900/90 border border-slate-800 p-4 rounded-2xl flex flex-col justify-between hover:border-amber-500/50 transition-colors shadow-lg">
                <div className="flex items-center justify-between text-slate-400 mb-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wider">Spec. Alpha (α)</span>
                    <ShieldCheck className="w-4 h-4 text-amber-400" />
                </div>
                <div>
                    <div className="text-2xl font-black text-white tracking-tight">
                        {(acceptanceRate * 100).toFixed(1)}
                        <span className="text-xs font-normal text-slate-400 ml-1">%</span>
                    </div>
                    <div className="text-[10px] text-amber-400 font-mono mt-0.5">Tree Verify Rate</div>
                </div>
            </div>
        </div>
    );
};