// dashboard/src/app/layout.tsx
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
    title: "NexusKV - Real-Time Paged KV-Cache Visualizer",
    description: "High-Throughput Paged KV-Cache & Speculative Inference Engine",
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" className="dark">
            <body className="bg-slate-950 text-slate-100 antialiased selection:bg-emerald-500 selection:text-white">
                {children}
            </body>
        </html>
    );
}