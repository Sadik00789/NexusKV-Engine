// dashboard/src/app/page.tsx
"use client";

import React, { useState, useEffect } from "react";
import { VramBlockCanvas, BlockInspectData } from "@/components/VramBlockCanvas";
import { MetricsPanel } from "@/components/MetricsPanel";
import {
    Play,
    Loader2,
    RefreshCw,
    Terminal,
    Sparkles,
    Cpu,
    CheckCircle2,
    Layers,
    MessageSquare,
    Zap,
} from "lucide-react";

export default function Home() {
    const [prompt, setPrompt] = useState(
        "Explain the architecture of PagedAttention and speculative tree decoding with O(1) page rollbacks."
    );
    const [maxTokens, setMaxTokens] = useState(64);
    const [generatedText, setGeneratedText] = useState("");
    const [isGenerating, setIsGenerating] = useState(false);

    // Live Telemetry State
    const [sequenceId, setSequenceId] = useState<number | undefined>(undefined);
    const [tokensPerSec, setTokensPerSec] = useState(0);
    const [ttftMs, setTtftMs] = useState(0);
    const [itlMs, setItlMs] = useState(0);
    const [allocatedBlocks, setAllocatedBlocks] = useState(0);
    const [totalBlocks, setTotalBlocks] = useState(512);
    const [memoryRatio, setMemoryRatio] = useState(0);
    const [prefixCacheHitRatio, setPrefixCacheHitRatio] = useState(0.42);
    const [acceptanceRate, setAcceptanceRate] = useState(0.85);
    const [activeBlockIds, setActiveBlockIds] = useState<number[]>([]);
    const [blocksMetadata, setBlocksMetadata] = useState<BlockInspectData[] | undefined>(undefined);

    // Engine connection status
    const [engineOnline, setEngineOnline] = useState(false);

    // Periodic system metrics probe
    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                const res = await fetch("http://127.0.0.1:8080/v1/metrics");
                if (res.ok) {
                    const data = await res.json();
                    setEngineOnline(true);
                    setTotalBlocks(data.total_blocks);
                    setAllocatedBlocks(data.allocated_blocks);
                    setMemoryRatio(data.memory_usage_percent / 100.0);
                    setPrefixCacheHitRatio(data.prefix_cache_hit_ratio || 0.45);
                    setAcceptanceRate(data.speculative_acceptance_rate || 0.85);
                    if (data.block_inspect_info) {
                        setBlocksMetadata(data.block_inspect_info);
                    }
                } else {
                    setEngineOnline(false);
                }
            } catch {
                setEngineOnline(false);
            }
        };

        fetchMetrics();
        const interval = setInterval(fetchMetrics, 3000);
        return () => clearInterval(interval);
    }, []);

    const startGenerationStream = async () => {
        if (isGenerating) return;
        setIsGenerating(true);
        setGeneratedText("");

        try {
            const response = await fetch("http://127.0.0.1:8080/v1/generate/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt, max_tokens: maxTokens }),
            });

            if (!response.body) {
                setIsGenerating(false);
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split("\n");

                for (const line of lines) {
                    if (line.startsWith("data:")) {
                        const rawJson = line.replace("data:", "").trim();
                        if (!rawJson || rawJson === "[DONE]") continue;

                        try {
                            const data = JSON.parse(rawJson);
                            setSequenceId(data.sequence_id);
                            setGeneratedText((prev) => prev + data.token);
                            setTokensPerSec(data.tokens_per_sec);
                            setTtftMs(data.ttft_ms);
                            setItlMs(data.itl_ms);
                            setAllocatedBlocks(data.allocated_blocks);
                            setTotalBlocks(data.total_blocks);
                            setMemoryRatio(data.memory_usage_ratio);
                            setPrefixCacheHitRatio(data.prefix_cache_hit_ratio);
                            setAcceptanceRate(data.speculative_acceptance_rate);
                            if (data.active_block_ids) {
                                setActiveBlockIds(data.active_block_ids);
                            }

                            if (data.is_finished) {
                                setIsGenerating(false);
                            }
                        } catch (e) {
                            console.error("JSON parse error", e);
                        }
                    }
                }
            }
        } catch (err) {
            console.error("Failed to connect to NexusKV engine:", err);
            // Simulated local playback if backend is offline
            simulateLocalStream();
        } finally {
            setIsGenerating(false);
        }
    };

    const simulateLocalStream = async () => {
        setIsGenerating(true);
        setGeneratedText("");
        const words = [
            "NexusKV",
            "executes",
            "PagedAttention",
            "with",
            "lock-free",
            "continuous",
            "batching,",
            "Radix",
            "prefix",
            "caching,",
            "and",
            "O(1)",
            "speculative",
            "tree",
            "page",
            "rollbacks",
            "at",
            "hardware",
            "line-rate.",
        ];

        const seq = (sequenceId ?? 0) + 1;
        setSequenceId(seq);
        setTtftMs(18.4);

        for (let i = 0; i < Math.min(words.length, maxTokens); i++) {
            await new Promise((r) => setTimeout(r, 45));
            setGeneratedText((prev) => prev + words[i] + " ");
            setTokensPerSec(58.4 + Math.random() * 4);
            setItlMs(16.8 + Math.random() * 2);
            setAllocatedBlocks(Math.min(512, 18 + i * 2));
            setMemoryRatio((18 + i * 2) / 512);
            setPrefixCacheHitRatio(0.68);
            setAcceptanceRate(0.88);
            setActiveBlockIds(Array.from({ length: 4 + Math.floor(i / 8) }, (_, idx) => idx * 3 + (seq % 4)));
        }
        setIsGenerating(false);
    };

    return (
        <main className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8">
            <div className="max-w-7xl mx-auto space-y-6">
                {/* Top Header */}
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800/80 pb-5">
                    <div>
                        <div className="flex items-center gap-3">
                            <span className="w-3.5 h-3.5 rounded-full bg-emerald-500 animate-pulse shadow-md shadow-emerald-500/50"></span>
                            <h1 className="text-2xl md:text-3xl font-black tracking-tight text-white flex items-center gap-2">
                                NexusKV <span className="text-emerald-400">Engine</span>
                            </h1>
                            <span className="px-2.5 py-0.5 bg-slate-900 border border-slate-700/80 rounded-full text-[11px] font-mono text-emerald-400">
                                Rust 2024 + CUDA 13.1
                            </span>
                        </div>
                        <p className="text-xs text-slate-400 mt-1">
                            High-Throughput Paged KV-Cache, Radix Prefix Caching & Speculative Tree Decodes
                        </p>
                    </div>

                    {/* Server status pill */}
                    <div className="flex items-center gap-3">
                        <span
                            className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono border ${
                                engineOnline
                                    ? "bg-emerald-950/80 border-emerald-700 text-emerald-300"
                                    : "bg-slate-900 border-slate-800 text-slate-400"
                            }`}
                        >
                            <span
                                className={`w-2 h-2 rounded-full ${
                                    engineOnline ? "bg-emerald-400 animate-ping" : "bg-slate-500"
                                }`}
                            ></span>
                            {engineOnline ? "Axum Engine Online (:8080)" : "Standby / Local Simulator"}
                        </span>
                    </div>
                </div>

                {/* Real-Time Metrics Panel (HUD) */}
                <MetricsPanel
                    tokensPerSec={tokensPerSec}
                    ttftMs={ttftMs}
                    itlMs={itlMs}
                    allocatedBlocks={allocatedBlocks}
                    totalBlocks={totalBlocks}
                    memoryRatio={memoryRatio}
                    prefixCacheHitRatio={prefixCacheHitRatio}
                    acceptanceRate={acceptanceRate}
                />

                {/* VRAM Physical Block Heatmap */}
                <VramBlockCanvas
                    allocatedBlocks={allocatedBlocks}
                    totalBlocks={totalBlocks}
                    acceptanceRate={acceptanceRate}
                    activeSequenceId={sequenceId}
                    activeBlockIds={activeBlockIds}
                    blocksMetadata={blocksMetadata}
                />

                {/* Generation Console & OpenAI API Playground */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Prompt Input Section */}
                    <div className="bg-slate-900/90 border border-slate-800 p-5 rounded-2xl flex flex-col justify-between shadow-xl">
                        <div className="space-y-4">
                            <div className="flex items-center justify-between">
                                <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                                    <Terminal className="w-4 h-4 text-emerald-400" />
                                    Inference Prompt
                                </label>
                                <span className="text-[11px] font-mono text-slate-400">
                                    Model: nexuskv-engine-v1
                                </span>
                            </div>

                            <textarea
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                rows={4}
                                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-white focus:outline-none focus:border-emerald-500 transition-colors placeholder:text-slate-600 font-sans"
                                placeholder="Enter prompt to execute continuous speculative decoding..."
                            />

                            <div className="flex items-center justify-between text-xs text-slate-400 pt-1">
                                <span className="font-mono">Max Tokens: {maxTokens}</span>
                                <input
                                    type="range"
                                    min="16"
                                    max="256"
                                    step="16"
                                    value={maxTokens}
                                    onChange={(e) => setMaxTokens(Number(e.target.value))}
                                    className="accent-emerald-500 w-48"
                                />
                            </div>
                        </div>

                        <div className="pt-4 flex items-center gap-3">
                            <button
                                onClick={startGenerationStream}
                                disabled={isGenerating}
                                className="flex-1 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 text-white font-semibold py-2.5 rounded-xl transition-all flex items-center justify-center gap-2 text-sm shadow-lg shadow-emerald-900/30"
                            >
                                {isGenerating ? (
                                    <>
                                        <Loader2 className="w-4 h-4 animate-spin" /> Streaming Tokens...
                                    </>
                                ) : (
                                    <>
                                        <Play className="w-4 h-4 fill-white" /> Trigger Speculative Stream
                                    </>
                                )}
                            </button>
                        </div>
                    </div>

                    {/* Decoded Stream Console */}
                    <div className="bg-slate-900/90 border border-slate-800 p-5 rounded-2xl flex flex-col shadow-xl">
                        <div className="flex items-center justify-between mb-3">
                            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                                <MessageSquare className="w-4 h-4 text-cyan-400" />
                                Decoded Stream Output
                            </label>
                            {isGenerating && (
                                <span className="text-xs text-emerald-400 font-mono animate-pulse flex items-center gap-1">
                                    <RefreshCw className="w-3 h-3 animate-spin" /> Decoding Iteration
                                </span>
                            )}
                        </div>

                        <div className="flex-1 bg-slate-950 border border-slate-800 rounded-xl p-4 font-mono text-sm text-emerald-300/90 overflow-y-auto max-h-[170px] min-h-[120px] whitespace-pre-wrap leading-relaxed">
                            {generatedText || (
                                <span className="text-slate-600 italic">
                                    Token output stream and real-time physical VRAM block mappings will stream here...
                                </span>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </main>
    );
}