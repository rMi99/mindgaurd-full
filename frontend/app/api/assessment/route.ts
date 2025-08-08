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
        scores: data.phq9  // Frontend sends as {"1": 0, "2": 1, etc.}
      },
      sleep: data.sleep,
      language: data.language || "en"
    }

    // Call backend API
    const response = await fetch(`${BACKEND_URL}/assessment`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(transformedData),
    })

    if (!response.ok) {
      const errorData = await response.json()
      return NextResponse.json({ error: errorData.detail || "Assessment processing failed" }, { status: response.status })
    }

    const result = await response.json()
    return NextResponse.json(result)
  } catch (error) {
    console.error("Assessment API error:", error)
    return NextResponse.json({ error: "Assessment processing failed" }, { status: 500 })
  }
}
