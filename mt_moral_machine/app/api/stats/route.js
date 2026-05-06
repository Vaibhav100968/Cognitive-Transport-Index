import { NextResponse } from "next/server";
import { FIXED_GLOBAL_STATS } from "@/data/fixedGlobalStats";

/** Fixed reference stats — same payload on every request (fixed-run build). */
export async function GET() {
  return NextResponse.json(FIXED_GLOBAL_STATS);
}
