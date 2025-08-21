import { type NextRequest, NextResponse } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"

export async function POST(request: NextRequest) {
  try {
    const data = await request.json()

    // Check for authentication
    const headerAuth = request.headers.get("authorization")
    const cookieToken = request.cookies.get("access_token")?.value
    const auth = headerAuth || (cookieToken ? `Bearer ${cookieToken}` : undefined)

    if (!auth) {
      return NextResponse.json({ error: "Authentication required" }, { status: 401 })
    }

    // Validate required data structure
    if (!data.demographics || !data.phq9 || !data.sleep) {
      return NextResponse.json({ 
        error: "Missing required assessment data",
        details: {
          demographics: !data.demographics ? "Missing demographics" : null,
          phq9: !data.phq9 ? "Missing PHQ-9 data" : null,
          sleep: !data.sleep ? "Missing sleep data" : null
        }
      }, { status: 400 })
    }

    // Validate demographics structure
    if (!data.demographics.age || !data.demographics.gender) {
      return NextResponse.json({ 
        error: "Incomplete demographics data",
        details: {
          age: !data.demographics.age ? "Age is required" : null,
          gender: !data.demographics.gender ? "Gender is required" : null
        }
      }, { status: 400 })
    }

    // Validate PHQ-9 structure - frontend sends PHQ-9 data directly as {"1": 0, "2": 1, ...}
    if (!data.phq9 || typeof data.phq9 !== 'object') {
      return NextResponse.json({ 
        error: "Invalid PHQ-9 data structure",
        details: "PHQ-9 data must be an object with question scores"
      }, { status: 400 })
    }

    // Check that we have at least some PHQ-9 responses
    const phq9Keys = Object.keys(data.phq9).filter(key => data.phq9[key] !== null && data.phq9[key] !== undefined)
    if (phq9Keys.length === 0) {
      return NextResponse.json({ 
        error: "Invalid PHQ-9 data",
        details: "At least one PHQ-9 question must be answered"
      }, { status: 400 })
    }

    // Validate sleep structure
    const requiredSleepFields = ["sleepHours", "sleepQuality", "exerciseFrequency", "stressLevel"]
    const missingSleepFields = requiredSleepFields.filter(field => !data.sleep[field])
    if (missingSleepFields.length > 0) {
      return NextResponse.json({ 
        error: "Incomplete sleep data",
        details: {
          missingFields: missingSleepFields
        }
      }, { status: 400 })
    }

    // Transform PHQ-9 data to match backend expectations
    const transformedData = {
      demographics: data.demographics,
      phq9: {
        scores: data.phq9 // Frontend sends as {"1": 0, "2": 1, ...}, backend expects {scores: {...}}
      },
      sleep: data.sleep,
      language: data.language || "en"
    }

    // Call backend API
    const response = await fetch(`${BACKEND_URL}/api/assessment/submit`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: auth,
      },
      body: JSON.stringify(transformedData),
    })

    const text = await response.text()
    let payload: any
    try { 
      payload = text ? JSON.parse(text) : {} 
    } catch { 
      payload = { detail: text } 
    }

    if (!response.ok) {
      console.error("Backend submit failed:", response.status, payload)
      
      // Provide user-friendly error messages
      let userMessage = "Assessment processing failed"
      if (response.status === 400) {
        if (payload.detail && payload.detail.includes("PHQ-9")) {
          userMessage = "Please ensure all mental health questions are answered correctly"
        } else if (payload.detail && payload.detail.includes("demographics")) {
          userMessage = "Please complete all demographic information"
        } else if (payload.detail && payload.detail.includes("sleep")) {
          userMessage = "Please complete all sleep and lifestyle questions"
        } else {
          userMessage = payload.detail || payload.error || "Please check that all required fields are completed"
        }
      } else if (response.status === 401) {
        userMessage = "Authentication required. Please log in again."
      } else if (response.status >= 500) {
        userMessage = "Server error. Please try again in a few moments."
      }
      
      return NextResponse.json({ 
        error: userMessage,
        technicalDetails: payload.detail || payload.error,
        status: response.status
      }, { status: response.status })
    }

    // Return the comprehensive assessment result directly
    return NextResponse.json(payload)
  } catch (error: any) {
    console.error("Assessment API error:", error)
    return NextResponse.json({ 
      error: error?.message || "Assessment processing failed",
      details: "Internal server error"
    }, { status: 500 })
  }
}
