"use client";

import React, { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { Info, Layers, Database } from "lucide-react";

export interface BlockInspectData {
    block_id: number;
    ref_count: number;
    is_free: boolean;
    mapped_seq_id: number | null;
    is_prefix_cached: boolean;
    is_speculative: boolean;
}

interface VramBlockCanvasProps {
    allocatedBlocks: number;
    totalBlocks?: number;
    acceptanceRate: number;
    activeSequenceId?: number;
    activeBlockIds?: number[];
    blocksMetadata?: BlockInspectData[];
}

const SEQUENCE_COLORS: string[] = [
    "#10B981", // Seq 1: Emerald
    "#06B6D4", // Seq 2: Cyan
    "#8B5CF6", // Seq 3: Violet
    "#F59E0B", // Seq 4: Amber
    "#F43F5E", // Seq 5: Rose
    "#3B82F6", // Seq 6: Blue
    "#D946EF", // Seq 7: Fuchsia
    "#14B8A6", // Seq 8: Teal
];

export const VramBlockCanvas: React.FC<VramBlockCanvasProps> = ({
    allocatedBlocks,
    totalBlocks = 512,
    acceptanceRate,
    activeSequenceId,
    activeBlockIds = [],
    blocksMetadata,
}) => {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const containerRef = useRef<HTMLDivElement | null>(null);

    const [hoveredBlock, setHoveredBlock] = useState<BlockInspectData | null>(null);
    const [selectedBlock, setSelectedBlock] = useState<BlockInspectData | null>(null);
    const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);

    // Defensive mathematical sanitization
    const safeTotal = Math.max(1, totalBlocks || 512);
    const safeAllocated = Number.isFinite(allocatedBlocks)
        ? Math.min(safeTotal, Math.max(0, allocatedBlocks))
        : 0;

    const activePercent = ((safeAllocated / safeTotal) * 100).toFixed(1);

    const cols = 32;
    const rows = Math.ceil(safeTotal / cols);
    const blockSize = 14;
    const gap = 3;

    // Fast lookup set for active blocks
    const activeBlockSet = useMemo(() => new Set(activeBlockIds), [activeBlockIds]);

    // Helper to determine color for each physical block
    const getBlockColor = useCallback(
        (blockId: number, meta?: BlockInspectData): string => {
            if (meta) {
                if (meta.is_free) return "#1E293B"; // Slate-800
                if (meta.is_prefix_cached && meta.ref_count > 1) return "#6366F1"; // Indigo (Radix Prefix)
                if (meta.is_speculative) return "#F59E0B"; // Amber (Draft Speculation)
                if (meta.mapped_seq_id !== null) {
                    const colorIdx = Math.abs(meta.mapped_seq_id - 1) % SEQUENCE_COLORS.length;
                    return SEQUENCE_COLORS[colorIdx];
                }
                return "#10B981"; // Emerald
            }

            // Fallback heuristics during live SSE streaming
            if (activeBlockSet.has(blockId)) {
                if (activeSequenceId) {
                    const colorIdx = Math.abs(activeSequenceId - 1) % SEQUENCE_COLORS.length;
                    return SEQUENCE_COLORS[colorIdx];
                }
                return "#10B981";
            }

            if (blockId < safeAllocated) {
                return acceptanceRate > 0.8 ? "#10B981" : "#F59E0B";
            }

            return "#1E293B";
        },
        [safeAllocated, acceptanceRate, activeSequenceId, activeBlockSet]
    );

    // Render Canvas
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
        const width = cols * (blockSize + gap) - gap;
        const height = rows * (blockSize + gap) - gap;

        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;

        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, width, height);

        for (let i = 0; i < safeTotal; i++) {
            const col = i % cols;
            const row = Math.floor(i / cols);
            const x = col * (blockSize + gap);
            const y = row * (blockSize + gap);

            const meta = blocksMetadata ? blocksMetadata[i] : undefined;
            const color = getBlockColor(i, meta);

            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.roundRect(x, y, blockSize, blockSize, 2.5);
            ctx.fill();

            // Glow / Stroke if hovered or selected
            const isHovered = hoveredBlock?.block_id === i;
            const isSelected = selectedBlock?.block_id === i;

            if (isSelected) {
                ctx.strokeStyle = "#FFFFFF";
                ctx.lineWidth = 2;
                ctx.stroke();
            } else if (isHovered) {
                ctx.strokeStyle = "#38BDF8";
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        }
    }, [
        safeTotal,
        safeAllocated,
        acceptanceRate,
        hoveredBlock,
        selectedBlock,
        blocksMetadata,
        getBlockColor,
        cols,
        rows,
    ]);

    // Handle Mouse Move over Canvas
    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const col = Math.floor(mouseX / (blockSize + gap));
        const row = Math.floor(mouseY / (blockSize + gap));

        if (col >= 0 && col < cols && row >= 0 && row < rows) {
            const blockId = row * cols + col;
            if (blockId < safeTotal) {
                const isActive = activeBlockSet.has(blockId) || blockId < safeAllocated;
                const meta = blocksMetadata?.[blockId] ?? {
                    block_id: blockId,
                    ref_count: isActive ? 1 : 0,
                    is_free: !isActive,
                    mapped_seq_id: isActive
                        ? activeSequenceId ?? ((blockId % 3) + 1)
                        : null,
                    is_prefix_cached: isActive && blockId % 4 === 0,
                    is_speculative: false,
                };

                setHoveredBlock(meta);

                // Ensure tooltip stays within container bounds
                const posX = Math.min(e.clientX - rect.left + 15, rect.width - 220);
                const posY = Math.max(10, e.clientY - rect.top - 70);
                setTooltipPos({ x: Math.max(10, posX), y: posY });
                return;
            }
        }

        setHoveredBlock(null);
        setTooltipPos(null);
    };

    const handleMouseLeave = () => {
        setHoveredBlock(null);
        setTooltipPos(null);
    };

    const handleClick = () => {
        if (hoveredBlock) {
            setSelectedBlock(hoveredBlock);
        }
    };

    const activeMeta = selectedBlock || hoveredBlock;

    return (
        <div
            ref={containerRef}
            className="relative bg-slate-900/90 backdrop-blur-md p-5 rounded-2xl border border-slate-800 shadow-xl overflow-hidden"
        >
            {/* Header with Title & Legend */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <div>
                    <h3 className="text-white text-sm font-semibold tracking-wide flex items-center gap-2">
                        <Layers className="w-4 h-4 text-emerald-400" />
                        Physical VRAM Page Map
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                        {safeTotal} Physical Pages @ 16 tokens/block (Non-contiguous CUDA PagedAttention)
                    </p>
                </div>

                {/* Multi-Tenant Sequence Legend */}
                <div className="flex flex-wrap items-center gap-3 text-xs">
                    <span className="flex items-center gap-1.5 text-emerald-400 font-mono">
                        <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500 shadow-sm shadow-emerald-500/50"></span>
                        Seq 1 (Emerald)
                    </span>
                    <span className="flex items-center gap-1.5 text-cyan-400 font-mono">
                        <span className="w-2.5 h-2.5 rounded-sm bg-cyan-500 shadow-sm shadow-cyan-500/50"></span>
                        Seq 2 (Cyan)
                    </span>
                    <span className="flex items-center gap-1.5 text-violet-400 font-mono">
                        <span className="w-2.5 h-2.5 rounded-sm bg-violet-500 shadow-sm shadow-violet-500/50"></span>
                        Seq 3 (Violet)
                    </span>
                    <span className="flex items-center gap-1.5 text-indigo-400 font-mono">
                        <span className="w-2.5 h-2.5 rounded-sm bg-indigo-500 shadow-sm shadow-indigo-500/50"></span>
                        Prefix Cached
                    </span>
                    <span className="flex items-center gap-1.5 text-slate-400 font-mono">
                        <span className="w-2.5 h-2.5 rounded-sm bg-slate-800"></span> Free
                    </span>
                </div>
            </div>

            {/* Canvas Container */}
            <div className="overflow-x-auto pb-2 flex justify-center">
                <canvas
                    ref={canvasRef}
                    onMouseMove={handleMouseMove}
                    onMouseLeave={handleMouseLeave}
                    onClick={handleClick}
                    className="cursor-crosshair rounded-lg transition-all"
                />
            </div>

            {/* Floating Tooltip */}
            {hoveredBlock && tooltipPos && (
                <div
                    style={{ left: `${tooltipPos.x}px`, top: `${tooltipPos.y}px` }}
                    className="pointer-events-none absolute z-30 bg-slate-950/95 border border-slate-700/80 rounded-xl p-3 shadow-2xl backdrop-blur-md text-xs space-y-1 w-52 animate-in fade-in zoom-in-95 duration-100"
                >
                    <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
                        <span className="font-bold text-white font-mono flex items-center gap-1">
                            <Database className="w-3 h-3 text-emerald-400" />
                            Block #{hoveredBlock.block_id}
                        </span>
                        <span
                            className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${hoveredBlock.is_free
                                    ? "bg-slate-800 text-slate-400"
                                    : hoveredBlock.is_prefix_cached
                                        ? "bg-indigo-950 text-indigo-300 border border-indigo-700/50"
                                        : "bg-emerald-950 text-emerald-300 border border-emerald-700/50"
                                }`}
                        >
                            {hoveredBlock.is_free
                                ? "Free Page"
                                : hoveredBlock.is_prefix_cached
                                    ? "Prefix Shared"
                                    : "Active Page"}
                        </span>
                    </div>

                    <div className="grid grid-cols-2 gap-1 pt-1 text-[11px] text-slate-300">
                        <span className="text-slate-400">Ref Count:</span>
                        <span className="font-mono font-bold text-white text-right">
                            {hoveredBlock.ref_count}
                        </span>

                        <span className="text-slate-400">Mapped Seq:</span>
                        <span className="font-mono text-right text-emerald-400">
                            {hoveredBlock.mapped_seq_id !== null
                                ? `Seq #${hoveredBlock.mapped_seq_id}`
                                : "Unmapped"}
                        </span>

                        <span className="text-slate-400">Speculative:</span>
                        <span className="font-mono text-right text-amber-400">
                            {hoveredBlock.is_speculative ? "Drafting" : "Committed"}
                        </span>
                    </div>
                </div>
            )}

            {/* Bottom Block Inspector HUD Bar */}
            <div className="mt-3 pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-400">
                <div className="flex items-center gap-2">
                    <Info className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                    <span className="truncate max-w-md">
                        {activeMeta
                            ? `Selected Block #${activeMeta.block_id}: Ref Count = ${activeMeta.ref_count
                            }, Mapped = ${activeMeta.mapped_seq_id !== null
                                ? `Sequence #${activeMeta.mapped_seq_id}`
                                : "None"
                            }, Status = ${activeMeta.is_free
                                ? "Free"
                                : activeMeta.is_prefix_cached
                                    ? "Radix Cached"
                                    : "Active"
                            }`
                            : "Hover or click any block to inspect VRAM physical page allocation."}
                    </span>
                </div>

                <div className="flex items-center gap-3 shrink-0">
                    <span className="font-mono text-emerald-400 font-semibold">
                        {safeAllocated} / {safeTotal} Blocks Active ({activePercent}%)
                    </span>
                </div>
            </div>
        </div>
    );
};