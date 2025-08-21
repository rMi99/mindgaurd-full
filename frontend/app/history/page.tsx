"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Calendar, TrendingUp, TrendingDown, Activity } from "lucide-react"
import Header from "../components/Header"
import Footer from "../components/Footer"

interface Assessment {
  id: string
  date: string
  phq9Score: number
  riskLevel: "low" | "moderate" | "high"
  sleepHours: string
  sleepQuality: string
}

interface HistoryData {
  assessments: Assessment[]
  totalAssessments?: number
  averageScore?: number
  latestScore?: number
  insights?: string[]
  trends?: {
    overallTrend: string
    direction: string
    recentAverage: number
    previousAverage: number
  }
}

export default function HistoryPage() {
  const [historyData, setHistoryData] = useState<HistoryData>({ assessments: [] })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeView, setActiveView] = useState<"all" | "stats" | "insights">("all")

  useEffect(() => {
    fetchHistoryData(activeView)
  }, [activeView])

  const fetchHistoryData = async (action: string = "all") => {
    try {
      setLoading(true)
      setError(null)
      
      const accessToken = localStorage.getItem('access_token')
      const userId = localStorage.getItem('user_id')
      
      if (!accessToken || !userId) {
        setError("Authentication required")
        return
      }

      const response = await fetch(`/api/history?userId=${userId}&action=${action}`, {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json'
        }
      })

      if (!response.ok) {
        throw new Error(`Backend responded with status: ${response.status}`)
      }

      const result = await response.json()
      
      // Handle wrapped and direct responses safely
      if (result && result.data) {
        setHistoryData({
          assessments: Array.isArray(result.data.assessments) ? result.data.assessments : Array.isArray(result.data) ? result.data : [],
          totalAssessments: result.data.totalAssessments,
          averageScore: result.data.averageScore,
          latestScore: result.data.latestScore,
          insights: result.data.insights,
          trends: result.data.trends
        })
      } else if (Array.isArray(result)) {
        setHistoryData({ assessments: result })
      } else {
        setHistoryData({ assessments: [] })
      }
    } catch (error: any) {
      console.error("Error fetching history data:", error)
      setError(error.message || "Failed to load history data")
      setHistoryData({ assessments: [] })
    } finally {
      setLoading(false)
    }
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

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "improving":
        return <TrendingUp className="h-4 w-4 text-green-600" />
      case "declining":
        return <TrendingDown className="h-4 w-4 text-red-600" />
      default:
        return <Activity className="h-4 w-4 text-blue-600" />
    }
  }

  const formatDate = (dateString: string): string => {
    try {
      return new Date(dateString).toLocaleDateString()
    } catch {
      return "Invalid Date"
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-lg text-gray-600">Loading history...</p>
          </div>
        </main>
        <Footer language="en" />
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-1 container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-900 mb-8">Assessment History</h1>
          
          {error && (
            <Alert className="mb-6 border-red-200 bg-red-50">
              <AlertDescription className="text-red-800">
                {error}
              </AlertDescription>
            </Alert>
          )}
          
          {/* View Selector */}
          <div className="mb-6">
            <div className="flex space-x-4">
              <Button
                variant={activeView === "all" ? "default" : "outline"}
                onClick={() => setActiveView("all")}
              >
                All Assessments
              </Button>
              <Button
                variant={activeView === "stats" ? "default" : "outline"}
                onClick={() => setActiveView("stats")}
              >
                Statistics
              </Button>
              <Button
                variant={activeView === "insights" ? "default" : "outline"}
                onClick={() => setActiveView("insights")}
              >
                Insights
              </Button>
            </div>
          </div>

          {/* Statistics View */}
          {activeView === "stats" && historyData.totalAssessments !== undefined && (
            <div className="grid md:grid-cols-3 gap-6 mb-8">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Total Assessments</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-3xl font-bold">{historyData.totalAssessments}</p>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Average Score</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-3xl font-bold">{historyData.averageScore?.toFixed(1) || "0"}</p>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Latest Score</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-3xl font-bold">{historyData.latestScore || "0"}</p>
                </CardContent>
              </Card>
            </div>
          )}

          {/* Insights View */}
          {activeView === "insights" && (
            <div className="space-y-6 mb-8">
              {historyData.insights && historyData.insights.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Insights</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2">
                      {historyData.insights.map((insight, index) => (
                        <li key={index} className="flex items-start space-x-2">
                          <span className="text-blue-600">•</span>
                          <span>{insight}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}
              
              {historyData.trends && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      {getTrendIcon(historyData.trends.direction)}
                      <span>Trends</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="mb-2">
                      <strong>Overall Trend:</strong> {historyData.trends.overallTrend}
                    </p>
                    <p className="mb-2">
                      <strong>Recent Average:</strong> {historyData.trends.recentAverage}
                    </p>
                    <p>
                      <strong>Previous Average:</strong> {historyData.trends.previousAverage}
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          )}

          {/* Assessments List */}
          <Card>
            <CardHeader>
              <CardTitle>Assessment History</CardTitle>
              <CardDescription>
                Your complete assessment record
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!Array.isArray(historyData.assessments) || historyData.assessments.length === 0 ? (
                <div className="text-center py-8">
                  <Calendar className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">No Assessments Yet</h3>
                  <p className="text-gray-600 mb-4">
                    Start your mental health journey by taking your first assessment.
                  </p>
                  <Button onClick={() => window.location.href = '/'}>
                    Take Assessment
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  {historyData.assessments.map((assessment) => (
                    <div key={assessment.id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center space-x-4">
                        <Calendar className="h-5 w-5 text-gray-400" />
                        <div>
                          <p className="font-medium">{formatDate(assessment.date)}</p>
                          <p className="text-sm text-gray-600">
                            PHQ-9: {assessment.phq9Score}/27 • Sleep: {assessment.sleepHours} • Quality: {assessment.sleepQuality}
                          </p>
                        </div>
                      </div>
                      <Badge className={getRiskLevelColor(assessment.riskLevel)}>
                        {assessment.riskLevel}
                      </Badge>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
          
          <div className="mt-8 text-center">
            <Button onClick={() => window.location.href = '/dashboard'}>
              Back to Dashboard
            </Button>
          </div>
        </div>
      </main>
      
      <Footer language="en" />
    </div>
  )
}
