import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"

export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization')
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return NextResponse.json({ error: 'Authorization header missing' }, { status: 401 })
    }

    const token = authHeader.split(' ')[1]

    const response = await fetch(`${BACKEND_URL}/profile`, {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      let errorData
      try {
        errorData = await response.json()
      } catch {
        errorData = { detail: "Unknown error" }
      }
      console.error("Backend profile error:", errorData)
      
      return NextResponse.json({ 
        error: errorData.detail || "Failed to fetch profile data",
        status: "error"
      }, { status: response.status })
    }

    const profileData = await response.json()
    
    return NextResponse.json({
      status: "success",
      data: profileData
    })
  } catch (error) {
    console.error('Profile API error:', error)
    return NextResponse.json({ 
      error: 'Failed to fetch profile data',
      status: "error"
    }, { status: 500 })
  }
}

export async function PUT(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization')
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return NextResponse.json({ error: 'Authorization header missing' }, { status: 401 })
    }

    const token = authHeader.split(' ')[1]
    const updateData = await request.json()

    const response = await fetch(`${BACKEND_URL}/profile`, {
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(updateData),
    })

    if (!response.ok) {
      let errorData
      try {
        errorData = await response.json()
      } catch {
        errorData = { detail: "Unknown error" }
      }
      console.error("Backend profile update error:", errorData)
      
      return NextResponse.json({ 
        error: errorData.detail || "Failed to update profile",
        status: "error"
      }, { status: response.status })
    }

    const updatedProfile = await response.json()
    
    return NextResponse.json({
      status: "success",
      data: updatedProfile,
      message: "Profile updated successfully"
    })
  } catch (error) {
    console.error('Profile update API error:', error)
    return NextResponse.json({ 
      error: 'Failed to update profile',
      status: "error"
    }, { status: 500 })
  }
}
