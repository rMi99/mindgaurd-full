import { type NextRequest, NextResponse } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const userId = searchParams.get("userId")
    const accessToken = searchParams.get("access_token")

    if (!userId || !accessToken) {
      return NextResponse.json({ error: "userId and access_token required" }, { status: 400 })
    }

    // Call backend with Authorization header
    const response = await fetch(`${BACKEND_URL}/dashboard`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
      },
    })
console.log("Response status:", response.status)
    if (!response.ok) {
      const errorData = await response.json()
      return NextResponse.json({ error: errorData.detail || "Failed to fetch dashboard data" }, { status: response.status })
    }

    const result = await response.json()
    return NextResponse.json(result)
  } catch (error) {
    console.error("Dashboard API error:", error)
    return NextResponse.json({ error: "Failed to fetch dashboard data" }, { status: 500 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const data = await request.json()
    const { userId, assessment, access_token } = data

    if (!userId || !assessment || !access_token) {
      return NextResponse.json({ error: "Missing required data" }, { status: 400 })
    }

    const response = await fetch(`${BACKEND_URL}/dashboard`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${access_token}`,
      },
      body: JSON.stringify({ userId, assessment }),
    })

    if (!response.ok) {
      const errorData = await response.json()
      return NextResponse.json({ error: errorData.detail || "Failed to save assessment" }, { status: response.status })
    }

    const result = await response.json()
    return NextResponse.json(result)
  } catch (error) {
    console.error("Dashboard POST error:", error)
    return NextResponse.json({ error: "Failed to save assessment" }, { status: 500 })
  }
}
