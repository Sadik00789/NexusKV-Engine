// dashboard/src/app/page.tsx
"use client";

import React, { useState, useEffect } from "react";
import { VramBlockCanvas, BlockInspectData } from "@/components/VramBlockCanvas";
import { MetricsPanel } from "@/components/MetricsPanel";
import { PipelinePanel, type PipelineMetrics } from "@/components/PipelinePanel";
import {
    applyReplay,
    countAllocated,
    type ReplayPayload,
} from "@/lib/replay";
import {
    Play,
    Loader2,
    RefreshCw,
    Terminal,
    MessageSquare,
    GitBranch,
    History,
} from "lucide-react";

const ENGINE_URL = "http://127.0.0.1:8080";

interface SessionTreeNode {
    id: number;
    parent: number | null;
    label: string;
    tokens: number;
    pinned_blocks: number;
}

function nodeDepth(node: SessionTreeNode, nodes: SessionTreeNode[]): number {
    let depth = 0;
    let parent = node.parent;
    while (parent !== null && depth < 32) {
        depth += 1;
        parent = nodes.find((candidate) => candidate.id === parent)?.parent ?? null;
    }
    return depth;
}

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
    const [blockSize, setBlockSize] = useState(16);
    const [memoryRatio, setMemoryRatio] = useState(0);
    const [prefixCacheHitRatio, setPrefixCacheHitRatio] = useState(0);
    const [acceptanceRate, setAcceptanceRate] = useState(0);
    const [activeBlockIds, setActiveBlockIds] = useState<number[]>([]);
    const [blocksMetadata, setBlocksMetadata] = useState<BlockInspectData[] | undefined>(undefined);
    const [executor, setExecutor] = useState<string>("unknown");
    const [pipeline, setPipeline] = useState<PipelineMetrics>({
        prefillTokens: 0,
        prefixSkipTokens: 0,
        adaptiveK: 0,
        evictions: 0,
        pageFaults: 0,
        swapBytesOut: 0,
        swapBytesIn: 0,
        attentionBackend: "unknown",
    });

    // Time-travel + sessions
    const [latestTick, setLatestTick] = useState(0);
    const [scrubTick, setScrubTick] = useState<number | null>(null);
    const [replayBlocks, setReplayBlocks] = useState<BlockInspectData[] | undefined>(undefined);
    const [sessionName, setSessionName] = useState("main");
    const [nodeId, setNodeId] = useState<number | null>(null);
    const [treeNodes, setTreeNodes] = useState<SessionTreeNode[]>([]);
    const [sessionBusy, setSessionBusy] = useState(false);
    const [sessionError, setSessionError] = useState("");

    // Engine connection status
    const [engineOnline, setEngineOnline] = useState(false);

    // Periodic system metrics probe
    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                const res = await fetch(`${ENGINE_URL}/v1/metrics`);
                if (res.ok) {
                    const data = await res.json();
                    setEngineOnline(true);
                    setExecutor(data.executor ?? "unknown");
                    setTotalBlocks(data.total_blocks);
                    if (typeof data.block_size === "number") {
                        setBlockSize(data.block_size);
                    }
                    setAllocatedBlocks(data.allocated_blocks);
                    setMemoryRatio(data.memory_usage_percent / 100.0);
                    setPrefixCacheHitRatio(Number(data.prefix_cache_hit_ratio ?? 0));
                    setAcceptanceRate(Number(data.speculative_acceptance_rate ?? 0));
                    setLatestTick(Number(data.event_tick ?? 0));
                    setPipeline({
                        prefillTokens: Number(data.prefill_tokens ?? 0),
                        prefixSkipTokens: Number(data.prefix_skip_tokens ?? 0),
                        adaptiveK: Number(data.adaptive_k ?? 0),
                        evictions: Number(data.evictions ?? 0),
                        pageFaults: Number(data.page_faults ?? 0),
                        swapBytesOut: Number(data.swap_bytes_out ?? 0),
                        swapBytesIn: Number(data.swap_bytes_in ?? 0),
                        attentionBackend: String(data.attention_backend ?? "unknown"),
                    });
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

    // Replay a historical tick when the scrubber is active. Live mode never
    // fabricates a replay: it simply uses the published metadata.
    useEffect(() => {
        if (scrubTick === null) {
            setReplayBlocks(undefined);
            return;
        }
        let cancelled = false;
        const replay = async () => {
            try {
                const res = await fetch(`${ENGINE_URL}/v1/timetravel?tick=${scrubTick}`);
                if (!res.ok) {
                    if (!cancelled) setReplayBlocks(undefined);
                    return;
                }
                const payload: ReplayPayload = await res.json();
                if (!cancelled) {
                    setReplayBlocks(applyReplay(payload.keyframe.blocks, payload.events));
                }
            } catch {
                if (!cancelled) setReplayBlocks(undefined);
            }
        };
        replay();
        return () => {
            cancelled = true;
        };
    }, [scrubTick]);

    const consumeGenerationStream = async (response: Response) => {
        if (!response.body) return;
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const frames = buffer.split("\n\n");
            buffer = frames.pop() ?? "";

            for (const frame of frames) {
                for (const line of frame.split("\n")) {
                    if (!line.startsWith("data:")) continue;
                    const rawJson = line.slice(5).trim();
                    if (!rawJson || rawJson === "[DONE]") continue;
                    try {
                        const data = JSON.parse(rawJson);
                        if (data.sequence_id !== undefined) setSequenceId(data.sequence_id);
                        const hasToken =
                            typeof data.token === "string" && data.token.length > 0;
                        if (typeof data.token === "string") {
                            setGeneratedText((prev) => prev + data.token);
                        }
                        // The terminal empty-token frame only reports a
                        // near-zero settling gap; keep the last token's measured
                        // timing instead of blanking the HUD after completion.
                        if (hasToken) {
                            const nextTps = Number(data.tokens_per_sec ?? 0);
                            if (nextTps > 0) setTokensPerSec(nextTps);
                            const nextTtft = Number(data.ttft_ms ?? 0);
                            if (nextTtft > 0) setTtftMs(nextTtft);
                            const nextItl = Number(data.itl_ms ?? 0);
                            if (nextItl > 0) setItlMs(nextItl);
                        }
                        setAllocatedBlocks(Number(data.allocated_blocks ?? 0));
                        setTotalBlocks(Number(data.total_blocks ?? 512));
                        setMemoryRatio(Number(data.memory_usage_ratio ?? 0));
                        setPrefixCacheHitRatio(Number(data.prefix_cache_hit_ratio ?? 0));
                        setAcceptanceRate(Number(data.speculative_acceptance_rate ?? 0));
                        setLatestTick(Number(data.event_tick ?? 0));
                        setPipeline((previous) => ({
                            ...previous,
                            prefillTokens: Number(data.prefill_tokens ?? previous.prefillTokens),
                            prefixSkipTokens: Number(
                                data.prefix_skip_tokens ?? previous.prefixSkipTokens
                            ),
                            adaptiveK: Number(data.adaptive_k ?? previous.adaptiveK),
                            attentionBackend: String(
                                data.attention_backend ?? previous.attentionBackend
                            ),
                        }));
                        if (Array.isArray(data.active_block_ids)) {
                            setActiveBlockIds(data.active_block_ids);
                        }
                        if (data.is_finished) {
                            setIsGenerating(false);
                        }
                    } catch {
                        // Ignore keep-alive or partial frames.
                    }
                }
            }
        }
    };

    const startGenerationStream = async () => {
        if (isGenerating) return;
        setIsGenerating(true);
        setGeneratedText("");

        try {
            const response = await fetch(`${ENGINE_URL}/v1/generate/stream`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt, max_tokens: maxTokens, spec: true }),
            });
            if (!response.ok) throw new Error(`engine returned ${response.status}`);
            await consumeGenerationStream(response);
        } catch (err) {
            console.error("Failed to connect to NexusKV engine:", err);
            setEngineOnline(false);
            setGeneratedText("Engine offline — no telemetry or token stream. Start the NexusKV server on :8080.");
        } finally {
            setIsGenerating(false);
        }
    };

    const fetchTree = async (session = sessionName) => {
        try {
            const res = await fetch(`${ENGINE_URL}/v1/sessions/${encodeURIComponent(session)}/tree`);
            if (!res.ok) return;
            const data = await res.json();
            setTreeNodes(Array.isArray(data.nodes) ? data.nodes : []);
        } catch {
            // Leave the previous tree visible when the engine is offline.
        }
    };

    const commitCheckpoint = async () => {
        if (sequenceId === undefined) {
            setSessionError("Generate at least one token before committing a checkpoint.");
            return;
        }
        setSessionBusy(true);
        setSessionError("");
        try {
            const res = await fetch(`${ENGINE_URL}/v1/sessions/commit`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session: sessionName,
                    parent: null,
                    label: `checkpoint-${sequenceId}`,
                    seq: sequenceId,
                }),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.error ?? `commit failed (${res.status})`);
            setNodeId(Number(data.node));
            await fetchTree();
        } catch (err) {
            setSessionError(err instanceof Error ? err.message : String(err));
        } finally {
            setSessionBusy(false);
        }
    };

    const checkoutNode = async () => {
        if (nodeId === null) return;
        setSessionBusy(true);
        setIsGenerating(true);
        setSessionError("");
        setGeneratedText("");
        try {
            const response = await fetch(`${ENGINE_URL}/v1/sessions/checkout`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    node: nodeId,
                    max_tokens: maxTokens,
                    temperature: 0.7,
                    spec: true,
                }),
            });
            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                throw new Error(data.error ?? `checkout failed (${response.status})`);
            }
            await consumeGenerationStream(response);
            await fetchTree();
        } catch (err) {
            setSessionError(err instanceof Error ? err.message : String(err));
        } finally {
            setIsGenerating(false);
            setSessionBusy(false);
        }
    };

    const forgetNode = async () => {
        if (nodeId === null) return;
        setSessionBusy(true);
        setSessionError("");
        try {
            const res = await fetch(`${ENGINE_URL}/v1/sessions/forget`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ node: nodeId }),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.error ?? `forget failed (${res.status})`);
            setNodeId(null);
            await fetchTree();
        } catch (err) {
            setSessionError(err instanceof Error ? err.message : String(err));
        } finally {
            setSessionBusy(false);
        }
    };

    const displayedBlocks = replayBlocks ?? blocksMetadata;
    const replayAllocated = replayBlocks ? countAllocated(replayBlocks) : 0;
    const displayedAllocated = replayBlocks ? replayAllocated : allocatedBlocks;
    const displayedMemoryRatio =
        replayBlocks && totalBlocks > 0 ? replayAllocated / totalBlocks : memoryRatio;

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
                                Rust 2024 · paged KV
                            </span>
                        </div>
                        <p className="text-xs text-slate-400 mt-1">
                            Paged KV-Cache, Radix Prefix Caching, Sessions &amp; Time Travel
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
                            {engineOnline
                                ? `Engine Online (:8080) · ${executor} · ${pipeline.attentionBackend}`
                                : "Engine Offline — no telemetry"}
                        </span>
                    </div>
                </div>

                {executor === "mock" && (
                    <div className="rounded-xl border border-amber-700/60 bg-amber-950/40 px-4 py-3 text-amber-200 text-sm">
                        <strong className="font-semibold">MOCK EXECUTOR</strong> — token streams are
                        synthetic and come from the deterministic mock model. No model weights are
                        loaded. This mode is labeled <code className="font-mono">executor: mock</code>{" "}
                        by <code className="font-mono">/v1/metrics</code> and{" "}
                        <code className="font-mono">/v1/models</code>.
                    </div>
                )}

                {/* Real-Time Metrics Panel (HUD) */}
                <MetricsPanel
                    tokensPerSec={tokensPerSec}
                    ttftMs={ttftMs}
                    itlMs={itlMs}
                    allocatedBlocks={displayedAllocated}
                    totalBlocks={totalBlocks}
                    memoryRatio={displayedMemoryRatio}
                    prefixCacheHitRatio={prefixCacheHitRatio}
                    acceptanceRate={acceptanceRate}
                />

                <PipelinePanel metrics={pipeline} />

                {/* Time-travel scrubber */}
                <div className="bg-slate-900/90 border border-slate-800 p-5 rounded-2xl shadow-xl">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
                        <div>
                            <h3 className="text-white text-sm font-semibold tracking-wide flex items-center gap-2">
                                <History className="w-4 h-4 text-cyan-400" />
                                Flight Recorder · Time Travel
                            </h3>
                            <p className="text-xs text-slate-400 mt-0.5">
                                {scrubTick === null
                                    ? "Live allocator state (null scrub tick)"
                                    : `Replaying keyframe + events up to tick ${scrubTick}`}
                            </p>
                        </div>
                        <div className="flex items-center gap-3">
                            <span
                                className={`px-2.5 py-1 rounded-full text-[11px] font-mono border ${
                                    scrubTick === null
                                        ? "bg-emerald-950/70 border-emerald-700 text-emerald-300"
                                        : "bg-cyan-950/70 border-cyan-700 text-cyan-300"
                                }`}
                            >
                                {scrubTick === null ? "LIVE" : `TICK ${scrubTick} / ${latestTick}`}
                            </span>
                            <button
                                onClick={() => setScrubTick(null)}
                                className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 border border-slate-700 transition-colors"
                            >
                                Return to Live
                            </button>
                        </div>
                    </div>
                    <input
                        type="range"
                        min={0}
                        max={Math.max(0, latestTick)}
                        value={scrubTick === null ? latestTick : Math.min(scrubTick, latestTick)}
                        disabled={!engineOnline || latestTick === 0}
                        onChange={(event) => setScrubTick(Number(event.target.value))}
                        className="w-full mt-4 accent-cyan-500 disabled:opacity-40"
                    />
                    <p className="text-[11px] text-slate-500 mt-1 font-mono">
                        Keyframes every 256 ticks · 50k-event ring buffer · scrubbing never fabricates state.
                    </p>
                </div>

                {/* VRAM Physical Block Heatmap */}
                <VramBlockCanvas
                    allocatedBlocks={displayedAllocated}
                    totalBlocks={totalBlocks}
                    blockSize={blockSize}
                    activeSequenceId={scrubTick === null ? sequenceId : undefined}
                    activeBlockIds={scrubTick === null ? activeBlockIds : []}
                    blocksMetadata={displayedBlocks}
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
                                    spec: on · adaptive K
                                </span>
                            </div>

                            <textarea
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                rows={4}
                                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-white focus:outline-none focus:border-emerald-500 transition-colors placeholder:text-slate-600 font-sans"
                                placeholder="Enter a prompt to run continuous speculative decoding..."
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
                                        <Play className="w-4 h-4 fill-white" /> Generate with Speculation
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

                        <div className="flex-1 bg-slate-950 border border-slate-800 rounded-xl p-4 font-mono text-sm text-emerald-300/90 overflow-y-auto max-h-[210px] min-h-[140px] whitespace-pre-wrap leading-relaxed">
                            {generatedText || (
                                <span className="text-slate-600 italic">
                                    Token output stream and real-time physical VRAM block mappings will stream here...
                                </span>
                            )}
                        </div>
                    </div>
                </div>

                {/* Git-for-KV sessions */}
                <div className="bg-slate-900/90 border border-slate-800 p-5 rounded-2xl shadow-xl">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
                        <div>
                            <h3 className="text-white text-sm font-semibold tracking-wide flex items-center gap-2">
                                <GitBranch className="w-4 h-4 text-violet-400" />
                                Git-for-KV Sessions
                            </h3>
                            <p className="text-xs text-slate-400 mt-0.5">
                                Commit pins the full-block KV prefix; checkout resumes it in O(1) and branches independently.
                            </p>
                        </div>
                        <div className="flex items-center gap-2">
                            <input
                                value={sessionName}
                                onChange={(event) => {
                                    setSessionName(event.target.value);
                                    setNodeId(null);
                                    setTreeNodes([]);
                                }}
                                className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs font-mono w-32 focus:outline-none focus:border-violet-500"
                                aria-label="Session name"
                            />
                            <button
                                onClick={commitCheckpoint}
                                disabled={isGenerating || sequenceId === undefined || sessionBusy}
                                className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-violet-700 hover:bg-violet-600 disabled:bg-slate-800 disabled:text-slate-500 transition-colors"
                            >
                                Commit
                            </button>
                            <button
                                onClick={checkoutNode}
                                disabled={nodeId === null || sessionBusy}
                                className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-cyan-700 hover:bg-cyan-600 disabled:bg-slate-800 disabled:text-slate-500 transition-colors"
                            >
                                Checkout
                            </button>
                            <button
                                onClick={forgetNode}
                                disabled={nodeId === null || sessionBusy}
                                className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 border border-slate-700 disabled:text-slate-500 transition-colors"
                            >
                                Forget
                            </button>
                        </div>
                    </div>

                    {sessionError && (
                        <p className="mt-3 text-xs text-rose-300 font-mono border border-rose-900/60 bg-rose-950/30 rounded-lg px-3 py-2">
                            {sessionError}
                        </p>
                    )}

                    <div className="mt-4">
                        {treeNodes.length === 0 ? (
                            <p className="text-xs text-slate-500 italic">
                                No checkpoints in this session. Run a generation with max-tokens above 1, then Commit.
                            </p>
                        ) : (
                            <div className="space-y-1 font-mono text-xs">
                                {treeNodes.map((node) => (
                                    <button
                                        key={node.id}
                                        onClick={() => setNodeId(node.id)}
                                        style={{ marginLeft: `${nodeDepth(node, treeNodes) * 18}px` }}
                                        className={`w-full text-left px-3 py-2 rounded-lg border transition-colors ${
                                            nodeId === node.id
                                                ? "border-violet-600 bg-violet-950/40 text-violet-200"
                                                : "border-slate-800 bg-slate-950/60 text-slate-300 hover:border-slate-700"
                                        }`}
                                    >
                                        <span className="text-violet-400">#{node.id}</span>{" "}
                                        <span>{node.label}</span>
                                        <span className="text-slate-500">
                                            {" "}· {node.tokens} tokens · {node.pinned_blocks} pinned blocks
                                        </span>
                                        {node.parent !== null && (
                                            <span className="text-slate-600"> · parent #{node.parent}</span>
                                        )}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </main>
    );
}
