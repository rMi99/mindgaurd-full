import { type NextRequest, NextResponse } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"

export async function POST(request: NextRequest) {
  try {
    const data = await request.json()

    if (!data.audio) {
      return NextResponse.json({ error: "Audio data is required" }, { status: 400 })
    }

    const response = await fetch(`${BACKEND_URL}/api/audio-analysis/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: "Unknown error" }))
      return NextResponse.json(
        { error: errorData.detail || errorData.error || "Failed to analyze audio" },
        { status: response.status }
      )
    }

    const result = await response.json()
    return NextResponse.json(result)
  } catch (error: any) {
    console.error("Audio analysis API error:", error)
    return NextResponse.json({ error: error.message || "Internal server error" }, { status: 500 })
  }
}

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/audio-analysis/health`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    })
    const result = await response.json()
    return NextResponse.json(result, { status: response.status })
  } catch (error: any) {
    console.error("Audio analysis health check error:", error)
    return NextResponse.json({ error: "Service unavailable" }, { status: 503 })
  }
}



