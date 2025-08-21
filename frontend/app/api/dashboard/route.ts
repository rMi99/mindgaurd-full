// API route for /api/dashboard
import { type NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const userId = searchParams.get("userId")
    const accessToken = searchParams.get("access_token")

    if (!userId) {
      return NextResponse.json({ error: "userId is required" }, { status: 400 })
    }

    // Build backend URL with user_id query parameter for temporary users
    const backendUrl = `${BACKEND_URL}/dashboard/?user_id=${encodeURIComponent(userId)}`
    
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    }

    // Add authorization header if available
    if (accessToken) {
      headers.Authorization = `Bearer ${accessToken}`
    }

    // Call backend with user_id in query params for temporary user support
    const response = await fetch(backendUrl, {
      method: "GET",
      headers,
    })

    console.log("Backend response status:", response.status)
    
    if (!response.ok) {
      let errorData
      try {
        errorData = await response.json()
      } catch {
        errorData = { detail: "Unknown error" }
      }
      console.error("Backend error:", errorData)
      return NextResponse.json({ error: errorData.detail || "Failed to fetch dashboard data" }, { status: response.status })
    }

    const result = await response.json()
    console.log("Backend result:", result)
    
    // Transform backend response to frontend expected format
    const dashboardData = {
      status: "success",
      userInfo: result.data?.userInfo || {
        id: userId,
        email: null,
        full_name: "Anonymous User",
        is_authenticated: false
      },
      history: result.data?.history || [],
      trends: result.data?.trends ? {
        overallTrend: "stable",
        phq9Trend: 0,
        sleepTrend: "stable",
        insights: ["Regular self-assessment helps build self-awareness"],
        recommendations: ["Continue monitoring your mental health regularly"]
      } : null,
      personalizedInsights: result.data?.personalizedInsights || {
        encouragingMessage: "Your mental health journey is important. Keep taking care of yourself.",
        psychologicalInsights: ["Regular self-assessment helps build self-awareness"],
        personalizedRecommendations: ["Continue monitoring your mental health regularly"],
        progressSummary: "You're taking positive steps for your mental health",
        nextSteps: ["Take your next assessment in 1 week"]
      },
      widgets: result.data?.widgets || {
        sleep_avg: 7.1,
        stress_avg: 4.2,
        exercise_per_week: 3,
        assessments_completed: 0
      }
    }

    return NextResponse.json(dashboardData)
  } catch (error) {
    console.error("Dashboard API error:", error)
    return NextResponse.json({ error: "Failed to fetch dashboard data" }, { status: 500 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const data = await request.json();
    // Accept both direct forward and structured payload
    const { userId, assessment, access_token } = data;
    let backendPayload = data;
    let headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (userId && assessment && access_token) {
      backendPayload = { userId, assessment };
      headers.Authorization = `Bearer ${access_token}`;
    } else if (request.headers.get('authorization')) {
      headers.authorization = request.headers.get('authorization') || '';
    }
    const response = await fetch(`${BACKEND_URL}/dashboard/`, {
      method: 'POST',
      headers,
      body: JSON.stringify(backendPayload),
    });
    const result = await response.json();
    if (!response.ok) {
      return NextResponse.json({ error: result.detail || result.error || 'Failed to save assessment' }, { status: response.status });
    }
    return NextResponse.json(result, { status: response.status });
  } catch (error) {
    console.error('Dashboard POST API error:', error);
    return NextResponse.json({ error: 'Dashboard endpoint not found' }, { status: 404 });
  }
}
