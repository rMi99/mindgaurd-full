"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts"
import { Calendar, TrendingUp, TrendingDown, AlertTriangle, Heart, Moon, Activity, User, LogOut, Settings } from "lucide-react"
import { useRouter } from "next/navigation"
import type { Language, AssessmentHistory, HistoricalTrend } from "@/lib/types"
import Header from "../components/Header"
import Footer from "../components/Footer"
import LanguageSelector from "../components/LanguageSelector"

interface PersonalizedInsights {
  encouragingMessage: string
  psychologicalInsights: string[]
  personalizedRecommendations: string[]
  progressSummary: string
  nextSteps: string[]
}

interface UserInfo {
  user_id: string
  email?: string
  username?: string
  is_temporary: boolean
}

export default function DashboardPage() {
  const router = useRouter()
  const [language, setLanguage] = useState<Language>("en")
  const [assessmentHistory, setAssessmentHistory] = useState<AssessmentHistory[]>([])
  const [trends, setTrends] = useState<HistoricalTrend | null>(null)
  const [personalizedInsights, setPersonalizedInsights] = useState<PersonalizedInsights | null>(null)
  const [userInfo, setUserInfo] = useState<UserInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    checkAuthAndFetchData()
  }, [])

  const checkAuthAndFetchData = async () => {
    try {
      setLoading(true)
      setError(null)

      // Check if user is authenticated or has temp user ID
      const accessToken = localStorage.getItem('access_token')
      const tempUserId = localStorage.getItem('temp_user_id') || localStorage.getItem("user_id")
      const guestMode = localStorage.getItem('mindguard_guest')
      
      if (!accessToken && !tempUserId && !guestMode) {
        // No authentication or guest mode, redirect to auth page
        router.push('/auth')
        return
      }

      await fetchDashboardData()
    } catch (error) {
      console.error("Failed to check auth and fetch data:", error)
      setError("Failed to load dashboard data")
    } finally {
      setLoading(false)
    }
  }

  // const fetchDashboardData = async () => {
  //   try {
  //     const accessToken = localStorage.getItem('access_token')
  //     const tempUserId = localStorage.getItem('temp_user_id') || localStorage.getItem("user_id")
      
  //     let url = '/api/dashboard'
  //     const headers: HeadersInit = {
  //       'Content-Type': 'application/json',
  //     }

  //     // Add authorization header if authenticated
  //     if (accessToken) {
  //       headers['Authorization'] = `Bearer ${accessToken}`
  //     } else if (tempUserId) {
  //       url += `?userId=${tempUserId}`
  //     }

  //     const response = await fetch(url, { headers })
      
  //     if (!response.ok) {
  //       if (response.status === 401) {
  //         // Token expired or invalid, redirect to auth
  //         localStorage.removeItem('access_token')
  //         localStorage.removeItem('user_id')
  //         localStorage.removeItem('user_email')
  //         localStorage.removeItem('username')
  //         router.push('/auth')
  //         return
  //       }
  //       throw new Error('Failed to fetch dashboard data')
  //     }

  //     const data = await response.json()
  //     setAssessmentHistory(data.history || [])
  //     setTrends(data.trends || null)
  //     setPersonalizedInsights(data.personalizedInsights || null)
  //     setUserInfo(data.userInfo || null)
  //   } catch (error) {
  //     console.error("Failed to fetch dashboard data:", error)
  //     setError("Failed to load dashboard data")
  //   }
  // }
const fetchDashboardData = async () => {
  try {
    const accessToken = localStorage.getItem('access_token')
    const userId = localStorage.getItem('user_id') || localStorage.getItem('temp_user_id')
    const guestMode = localStorage.getItem('mindguard_guest')

    // For guest mode, create a temporary user ID
    if (guestMode && !userId) {
      const tempId = `guest_${Date.now()}`
      localStorage.setItem('temp_user_id', tempId)
    }

    const finalUserId = userId || localStorage.getItem('temp_user_id') || 'guest_user'
    
    let url = `/api/dashboard?userId=${encodeURIComponent(finalUserId)}`
    if (accessToken) {
      url += `&access_token=${encodeURIComponent(accessToken)}`
    }

    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      }
    })

    if (!response.ok) {
      if (response.status === 401 && accessToken) {
        // Only clear auth data if we had a token that was rejected
        localStorage.removeItem('access_token')
        localStorage.removeItem('user_id')
        localStorage.removeItem('user_email')
        localStorage.removeItem('username')
        router.push('/auth')
        return
      }
      throw new Error('Failed to fetch dashboard data')
    }

    const responseData = await response.json()
    
    // Extract data from the backend response structure
    const backendData = responseData.data || responseData
    
    // Set the assessment history - transform backend format to frontend format
    const historyData = backendData.history || []
    const transformedHistory = historyData.map((assessment: any) => ({
      id: assessment.id,
      phq9Score: assessment.phq9_score || 0,
      riskLevel: assessment.risk_level || backendData.current_risk_level || "unknown",
      date: assessment.created_at || new Date().toISOString(),
      sleepHours: assessment.sleep_hours || 0,
      sleepQuality: assessment.sleep_quality || "unknown"
    }))
    
    // If no history, create a summary entry from current status
    if (transformedHistory.length === 0 && backendData.current_risk_level) {
      transformedHistory.push({
        id: 'current',
        phq9Score: 0,
        riskLevel: backendData.current_risk_level,
        date: new Date().toISOString(),
        sleepHours: Math.round(backendData.widgets?.sleep_avg || 7),
        sleepQuality: "good"
      })
    }
    
    // Transform trends data
    const trendsData = backendData.trends ? {
      overallTrend: "stable", // Default to stable
      phq9Trend: 0,
      sleepTrend: "stable",
      insights: ["Regular self-assessment helps build self-awareness"],
      recommendations: ["Continue monitoring your mental health regularly"],
      correlations: {}, // Required by HistoricalTrend interface
      data: backendData.trends // Keep the raw trend data for charts
    } : null
    
    setAssessmentHistory(transformedHistory)
    setTrends(trendsData)
    setPersonalizedInsights(backendData.personalizedInsights || null)
    setUserInfo(backendData.userInfo || { user_id: finalUserId, is_temporary: !accessToken })
  } catch (error) {
    console.error("Failed to fetch dashboard data:", error)
    setError("Failed to load dashboard data")
  }
}


  const handleLogout = () => {
    // Clear all authentication data
    localStorage.removeItem('access_token')
    localStorage.removeItem('user_id')
    localStorage.removeItem('user_email')
    localStorage.removeItem('username')
    localStorage.removeItem('temp_user_id')
    localStorage.removeItem('is_temporary_user')
    
    // Redirect to home
    router.push('/')
  }

  const handleSignUp = () => {
    router.push('/auth')
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <Header />
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600">Loading your dashboard...</p>
            </div>
          </div>
        </div>
        <Footer language={language} />
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <Header />
        <div className="container mx-auto px-4 py-8">
          <Alert className="max-w-md mx-auto border-red-200 bg-red-50">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription className="text-red-800">
              {error}
            </AlertDescription>
          </Alert>
          <div className="text-center mt-4">
            <Button onClick={() => window.location.reload()}>
              Try Again
            </Button>
          </div>
        </div>
        <Footer language={language} />
      </div>
    )
  }

  const getRiskLevelColor = (riskLevel: string) => {
    switch (riskLevel) {
      case "low":
        return "bg-green-100 text-green-800"
      case "moderate":
        return "bg-yellow-100 text-yellow-800"
      case "high":
        return "bg-red-100 text-red-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  const getTrendIcon = (trend: "improving" | "stable" | "declining") => {
    switch (trend) {
      case "improving":
        return <TrendingUp className="h-4 w-4 text-green-600" />
      case "declining":
        return <TrendingDown className="h-4 w-4 text-red-600" />
      default:
        return <Activity className="h-4 w-4 text-blue-600" />
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-lg text-gray-600">Loading your dashboard...</p>
          </div>
        </main>
        <Footer language={language} />
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 px-4 py-8">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8">
            <LanguageSelector currentLanguage={language} onLanguageChange={setLanguage} />
          </div>

          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Your Mental Health Dashboard</h1>
            <p className="text-gray-600">Track your progress and understand your mental health journey over time</p>
          </div>

          {assessmentHistory.length === 0 ? (
            <Card className="text-center py-12">
              <CardContent>
                <Heart className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Welcome to MindGuard</h3>
                <p className="text-gray-600 mb-6">
                  Start your mental health journey by taking your first assessment. Your progress will be tracked here
                  over time.
                </p>
                <Button asChild>
                  <a href="/">Take Your First Assessment</a>
                </Button>
              </CardContent>
            </Card>
          ) : (
            <Tabs defaultValue="overview" className="space-y-6">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="trends">Trends</TabsTrigger>
                <TabsTrigger value="history">History</TabsTrigger>
                <TabsTrigger value="insights">Insights</TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="space-y-6">
                {/* Current Status Cards */}
                <div className="grid md:grid-cols-3 gap-6">
                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Current Risk Level</CardTitle>
                      <AlertTriangle className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold mb-2">{assessmentHistory[0]?.riskLevel || "Unknown"}</div>
                      <Badge className={getRiskLevelColor(assessmentHistory[0]?.riskLevel || "")}>
                        PHQ-9: {assessmentHistory[0]?.phq9Score || 0}/27
                      </Badge>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Recent Trend</CardTitle>
                      {trends && getTrendIcon(trends.overallTrend as "improving" | "stable" | "declining")}
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold mb-2 capitalize">{trends?.overallTrend || "Stable"}</div>
                      <p className="text-xs text-muted-foreground">
                        Based on last {assessmentHistory.length} assessments
                      </p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Sleep Quality</CardTitle>
                      <Moon className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold mb-2 capitalize">
                        {assessmentHistory[0]?.sleepQuality || "Unknown"}
                      </div>
                      <p className="text-xs text-muted-foreground">
                        {assessmentHistory[0]?.sleepHours || "Unknown"} hours/night
                      </p>
                    </CardContent>
                  </Card>
                </div>

                {/* Quick Actions */}
                <Card>
                  <CardHeader>
                    <CardTitle>Quick Actions</CardTitle>
                    <CardDescription>Take action based on your current status</CardDescription>
                  </CardHeader>
                  <CardContent className="flex flex-wrap gap-4">
                    <Button asChild>
                      <a href="/">Take New Assessment</a>
                    </Button>
                    <Button variant="outline" asChild>
                      <a href="/history">View Detailed History</a>
                    </Button>
                    <Button variant="outline" asChild>
                      <a href="/resources">View Resources</a>
                    </Button>
                    <Button variant="outline" asChild>
                      <a href="/export">Export Data</a>
                    </Button>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="trends" className="space-y-6">
                {/* PHQ-9 Score Trend */}
                <Card>
                  <CardHeader>
                    <CardTitle>PHQ-9 Score Over Time</CardTitle>
                    <CardDescription>Track your depression screening scores</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={assessmentHistory.slice().reverse()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" tickFormatter={(value) => new Date(value).toLocaleDateString()} />
                        <YAxis domain={[0, 27]} />
                        <Tooltip
                          labelFormatter={(value) => new Date(value).toLocaleDateString()}
                          formatter={(value: any) => [value, "PHQ-9 Score"]}
                        />
                        <Line
                          type="monotone"
                          dataKey="phq9Score"
                          stroke="#3b82f6"
                          strokeWidth={2}
                          dot={{ fill: "#3b82f6", strokeWidth: 2, r: 4 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                {/* Sleep Hours Trend */}
                <Card>
                  <CardHeader>
                    <CardTitle>Sleep Patterns</CardTitle>
                    <CardDescription>Monitor your sleep duration and quality</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={assessmentHistory.slice().reverse()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" tickFormatter={(value) => new Date(value).toLocaleDateString()} />
                        <YAxis />
                        <Tooltip labelFormatter={(value) => new Date(value).toLocaleDateString()} />
                        <Bar dataKey="sleepHoursNumeric" fill="#10b981" />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="history" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Assessment History</CardTitle>
                    <CardDescription>Complete record of your assessments</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {assessmentHistory.map((assessment, index) => (
                        <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                          <div className="flex items-center space-x-4">
                            <Calendar className="h-5 w-5 text-gray-400" />
                            <div>
                              <p className="font-medium">{new Date(assessment.date).toLocaleDateString()}</p>
                              <p className="text-sm text-gray-600">
                                PHQ-9: {assessment.phq9Score}/27 • Sleep: {assessment.sleepHours}
                              </p>
                            </div>
                          </div>
                          <Badge className={getRiskLevelColor(assessment.riskLevel)}>{assessment.riskLevel}</Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="insights" className="space-y-6">
                {trends && (
                  <>
                    <Card>
                      <CardHeader>
                        <CardTitle>Personalized Insights</CardTitle>
                        <CardDescription>AI-generated insights based on your data</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        {trends.insights.map((insight, index) => (
                          <div key={index} className="p-4 bg-blue-50 rounded-lg">
                            <p className="text-sm text-blue-800">{insight}</p>
                          </div>
                        ))}
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader>
                        <CardTitle>Recommendations</CardTitle>
                        <CardDescription>Personalized suggestions for improvement</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        {trends.recommendations.map((recommendation, index) => (
                          <div key={index} className="flex items-start space-x-3">
                            <div className="flex-shrink-0 w-6 h-6 bg-green-100 rounded-full flex items-center justify-center mt-0.5">
                              <span className="text-xs font-medium text-green-600">{index + 1}</span>
                            </div>
                            <p className="text-sm text-gray-700">{recommendation}</p>
                          </div>
                        ))}
                      </CardContent>
                    </Card>
                  </>
                )}
              </TabsContent>
            </Tabs>
          )}
        </div>
      </main>

      <Footer language={language} />
    </div>
  )
}
