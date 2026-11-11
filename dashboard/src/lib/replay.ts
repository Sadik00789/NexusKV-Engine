// dashboard/src/lib/replay.ts
export interface ReplayBlock {
    block_id: number;
    ref_count: number;
    is_free: boolean;
    mapped_seq_id: number | null;
    is_prefix_cached: boolean;
    is_speculative: boolean;
    state?: string;
}

export interface MemEvent {
    tick: number;
    kind: "alloc" | "free" | "hit" | "fault" | "evict" | "fork" | "cow" | "rollback" | "pin";
    seq: number | null;
    blocks: number[];
    cached?: boolean;
}

export interface Keyframe {
    tick: number;
    blocks: ReplayBlock[];
}

export interface ReplayPayload {
    keyframe: Keyframe;
    events: MemEvent[];
}

/**
 * Apply flight-recorder events to a keyframe. Semantics mirror
 * `memory::events::replay_state` in the Rust engine:
 *
 *   alloc/hit/fork/pin -> ref++, active
 *   free               -> ref--, then cached (if event.cached) or free at zero
 *   evict              -> free
 *   cow/fault/rollback -> no-op (allocation/release events already cover them)
 */
export function applyReplay(
    keyframeBlocks: ReplayBlock[],
    events: MemEvent[]
): ReplayBlock[] {
    const blocks = keyframeBlocks.map((block) => ({ ...block }));
    for (const event of events) {
        for (const blockId of event.blocks) {
            const block = blocks[blockId];
            if (!block) continue;
            switch (event.kind) {
                case "alloc":
                case "hit":
                case "fork":
                case "pin":
                    block.ref_count += 1;
                    block.is_free = false;
                    block.state = "active";
                    break;
                case "free":
                    block.ref_count = Math.max(0, block.ref_count - 1);
                    if (block.ref_count === 0) {
                        block.is_free = !event.cached;
                        block.state = event.cached ? "cached" : "free";
                    } else {
                        block.is_free = false;
                        block.state = "active";
                    }
                    break;
                case "evict":
                    block.ref_count = 0;
                    block.is_free = true;
                    block.state = "free";
                    break;
                case "cow":
                case "fault":
                case "rollback":
                    break;
            }
        }
    }
    return blocks;
}

export function countAllocated(blocks: ReplayBlock[]): number {
    return blocks.filter((block) => !block.is_free).length;
}
