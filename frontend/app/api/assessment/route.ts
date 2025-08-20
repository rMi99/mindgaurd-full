import { type NextRequest, NextResponse } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"

export async function POST(request: NextRequest) {
  try {
    const data = await request.json()

    // Validate required data
    if (!data.demographics || !data.phq9 || !data.sleep) {
      return NextResponse.json({ error: "Missing required assessment data" }, { status: 400 })
    }

    // Transform PHQ-9 data to match backend expectations
    const transformedData = {
      demographics: data.demographics,
      phq9: {
        scores: data.phq9 // Frontend sends as {"1": 0, "2": 1, ...}
      },
      sleep: data.sleep,
      language: data.language || "en"
    }

    // Forward Authorization header if present; fallback to cookie for SSR-based flows
    const headerAuth = request.headers.get("authorization") || undefined
    const cookieToken = request.cookies.get("access_token")?.value
    const auth = headerAuth || (cookieToken ? `Bearer ${cookieToken}` : undefined)

    // Call backend API (explicit /api prefix)
    const response = await fetch(`${BACKEND_URL}/api/assessment/submit`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(auth ? { Authorization: auth } : {}),
      },
      body: JSON.stringify(transformedData),
    })

    const text = await response.text()
    let payload: any
    try { payload = text ? JSON.parse(text) : {} } catch { payload = { detail: text } }

    if (!response.ok) {
      console.error("Backend submit failed:", response.status, payload)
      return NextResponse.json({ error: payload.detail || payload.error || "Assessment processing failed" }, { status: response.status })
    }

    return NextResponse.json(payload)
  } catch (error: any) {
    console.error("Assessment API error:", error)
    return NextResponse.json({ error: error?.message || "Assessment processing failed" }, { status: 500 })
  }
}
