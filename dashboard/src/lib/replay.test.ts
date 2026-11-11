import { describe, expect, it } from "vitest";
import { applyReplay, countAllocated, type ReplayBlock } from "./replay";

function block(block_id: number, state: "free" | "active" | "cached", ref_count: number): ReplayBlock {
    return {
        block_id,
        ref_count,
        is_free: state === "free",
        mapped_seq_id: null,
        is_prefix_cached: state === "cached",
        is_speculative: false,
        state,
    };
}

describe("flight-recorder replay reducer", () => {
    it("tracks allocation, cache transitions, and eviction", () => {
        const keyframe = [block(0, "free", 0), block(1, "free", 0), block(2, "free", 0)];
        const blocks = applyReplay(keyframe, [
            { tick: 1, kind: "alloc", seq: 1, blocks: [0] },
            { tick: 2, kind: "alloc", seq: 1, blocks: [1] },
            { tick: 3, kind: "free", seq: 1, blocks: [1], cached: true },
            { tick: 4, kind: "hit", seq: 2, blocks: [0] },
            { tick: 5, kind: "evict", seq: null, blocks: [0] },
        ]);

        expect(blocks[0].state).toBe("free");
        expect(blocks[0].ref_count).toBe(0);
        expect(blocks[1].state).toBe("cached");
        expect(blocks[1].ref_count).toBe(0);
        expect(blocks[1].is_free).toBe(false);
        expect(blocks[2].state).toBe("free");
        expect(countAllocated(blocks)).toBe(1);
    });

    it("ignores bookkeeping-only events", () => {
        const keyframe = [block(0, "active", 2)];
        const blocks = applyReplay(keyframe, [
            { tick: 1, kind: "cow", seq: 1, blocks: [0] },
            { tick: 2, kind: "fault", seq: 1, blocks: [0] },
            { tick: 3, kind: "rollback", seq: 1, blocks: [0] },
        ]);
        expect(blocks[0].ref_count).toBe(2);
        expect(blocks[0].state).toBe("active");
    });
});
