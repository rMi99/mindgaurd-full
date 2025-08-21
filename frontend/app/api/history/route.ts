import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const userId = searchParams.get('userId')
    const action = searchParams.get('action')
    
    if (!userId) {
      return NextResponse.json(
        { error: 'User ID is required' },
        { status: 400 }
      )
    }

    let endpoint = `${BACKEND_URL}/history/${userId}`
    
    // Handle different actions
    if (action === 'stats') {
      endpoint = `${BACKEND_URL}/history/${userId}/stats`
    } else if (action === 'insights') {
      endpoint = `${BACKEND_URL}/history/${userId}/insights`
    } else {
      // Add query parameters for filtering
      const startDate = searchParams.get('startDate')
      const endDate = searchParams.get('endDate')
      const riskLevels = searchParams.get('riskLevels')
      const limit = searchParams.get('limit')
      
      const params = new URLSearchParams()
      if (startDate) params.append('start_date', startDate)
      if (endDate) params.append('end_date', endDate)
      if (riskLevels) params.append('risk_levels', riskLevels)
      if (limit) params.append('limit', limit)
      
      if (params.toString()) {
        endpoint += `?${params.toString()}`
      }
    }

    const authHeader = request.headers.get('authorization')
    
    const response = await fetch(endpoint, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': authHeader || '',
      },
    })

    if (!response.ok) {
      let errorData
      try {
        errorData = await response.json()
      } catch {
        errorData = { detail: "Unknown error" }
      }
      console.error(`Backend responded with status: ${response.status}`, errorData)
      
      // Return structured error response instead of throwing
      return NextResponse.json(
        { 
          error: errorData.detail || `Failed to fetch history data (${response.status})`,
          status: "error",
          statusCode: response.status
        },
        { status: response.status }
      )
    }

    const data = await response.json()
    return NextResponse.json({
      status: "success",
      data: data,
      action: action || "recent"
    })
  } catch (error) {
    console.error('Error fetching history data:', error)
    return NextResponse.json(
      { 
        error: 'Failed to fetch history data', 
        status: "error",
        details: error instanceof Error ? error.message : "Unknown error"
      },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const userId = searchParams.get('userId')
    const action = searchParams.get('action')
    
    if (!userId) {
      return NextResponse.json(
        { error: 'User ID is required' },
        { status: 400 }
      )
    }

    let endpoint = `${BACKEND_URL}/history/${userId}`
    let body = null

    if (action === 'export') {
      endpoint = `${BACKEND_URL}/history/${userId}/export`
    } else if (action === 'note') {
      const requestBody = await request.json()
      endpoint = `${BACKEND_URL}/history/${userId}/note`
      const params = new URLSearchParams()
      params.append('assessment_id', requestBody.assessmentId)
      params.append('note', requestBody.note)
      endpoint += `?${params.toString()}`
    }

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: body ? JSON.stringify(body) : undefined,
    })

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error processing history request:', error)
    return NextResponse.json(
      { error: 'Failed to process history request' },
      { status: 500 }
    )
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const userId = searchParams.get('userId')
    const confirm = searchParams.get('confirm')
    
    if (!userId) {
      return NextResponse.json(
        { error: 'User ID is required' },
        { status: 400 }
      )
    }

    const endpoint = `${BACKEND_URL}/history/${userId}?confirm=${confirm || 'false'}`

    const response = await fetch(endpoint, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error deleting history:', error)
    return NextResponse.json(
      { error: 'Failed to delete history' },
      { status: 500 }
    )
  }
}

