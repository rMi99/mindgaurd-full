import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  try {
    const data = await request.json()

    // Forward the request to the backend
    const response = await fetch(`${BACKEND_URL}/api/facial-dashboard/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: "Unknown error" }))
      return NextResponse.json(
        { error: errorData.detail || errorData.error || "Failed to analyze facial expression" },
        { status: response.status }
      )
    }

    const result = await response.json()
    return NextResponse.json(result)
    
  } catch (error: any) {
    console.error("Facial dashboard API error:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
