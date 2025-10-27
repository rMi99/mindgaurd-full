import { NextRequest, NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization')
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return NextResponse.json({ error: 'Authorization header missing' }, { status: 401 })
    }

    const token = authHeader.split(' ')[1]

    // Default settings - in production, this would come from backend
    const defaultSettings = {
      notifications: {
        email_notifications: true,
        push_notifications: true,
        assessment_reminders: true,
        weekly_reports: true,
        emergency_alerts: true,
        marketing_emails: false,
      },
      privacy: {
        data_sharing: false,
        analytics_tracking: true,
        profile_visibility: 'private',
        show_activity: false,
      },
      app: {
        theme: 'system',
        language: 'en',
        timezone: 'UTC',
        date_format: 'DD/MM/YYYY',
        currency: 'USD',
      },
    }

    // In production, fetch from backend
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/settings`, {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    })

    if (response.ok) {
      const settings = await response.json()
      return NextResponse.json(settings)
    } else {
      // Fallback to default settings if backend doesn't have settings endpoint
      return NextResponse.json(defaultSettings)
    }
  } catch (error) {
    console.error('Settings fetch error:', error)
    
    // Return default settings on error
    const defaultSettings = {
      notifications: {
        email_notifications: true,
        push_notifications: true,
        assessment_reminders: true,
        weekly_reports: true,
        emergency_alerts: true,
        marketing_emails: false,
      },
      privacy: {
        data_sharing: false,
        analytics_tracking: true,
        profile_visibility: 'private',
        show_activity: false,
      },
      app: {
        theme: 'system',
        language: 'en',
        timezone: 'UTC',
        date_format: 'DD/MM/YYYY',
        currency: 'USD',
      },
    }
    
    return NextResponse.json(defaultSettings)
  }
}

export async function PUT(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization')
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return NextResponse.json({ error: 'Authorization header missing' }, { status: 401 })
    }

    const token = authHeader.split(' ')[1]
    const settings = await request.json()

    // In production, save to backend
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/settings`, {
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(settings),
    })

    if (response.ok) {
      const savedSettings = await response.json()
      return NextResponse.json(savedSettings)
    } else {
      // Fallback - just return the settings as if saved
      return NextResponse.json(settings)
    }
  } catch (error) {
    console.error('Settings save error:', error)
    // For now, return the settings as if saved
    const settings = await request.json()
    return NextResponse.json(settings)
  }
}
