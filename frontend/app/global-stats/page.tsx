"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar } from "recharts"
import { Users, TrendingUp, Heart, Globe, Lightbulb, MessageCircle, Activity, Star } from "lucide-react"
import Header from "../components/Header"
import Footer from "../components/Footer"
import type { Language } from "@/lib/types"

interface GlobalStatistics {
  totalUsers: number
  totalRegisteredUsers: number
  totalTemporaryUsers: number
  totalAssessments: number
  averageRiskLevel: string
  riskLevelDistribution: { [key: string]: number }
  mentalHealthTrends: {
    trend_direction: string
    weekly_data: Array<{
      date: string
      averageScore: number
      assessmentCount: number
    }>
    insights: string[]
  }
  encouragingMessages: string[]
  psychologicalInsights: string[]
  platformInsights: string[]
}

interface FeelGoodContent {
  daily_affirmations: string[]
  mental_health_tips: string[]
  success_stories: string[]
  community_impact: string[]
}

export default function GlobalStatsPage() {
  const [language, setLanguage] = useState<Language>("en")
  const [globalStats, setGlobalStats] = useState<GlobalStatistics | null>(null)
  const [feelGoodContent, setFeelGoodContent] = useState<FeelGoodContent | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchGlobalData()
  }, [])

  const fetchGlobalData = async () => {
    try {
      setLoading(true)
      setError(null)

      // Fetch global statistics
      const statsResponse = await fetch('/api/global-stats')
      if (!statsResponse.ok) {
        throw new Error('Failed to fetch global statistics')
      }
      const statsData = await statsResponse.json()
      setGlobalStats(statsData)

      // Fetch feel-good content
      const contentResponse = await fetch('/api/global-stats/feel-good-content')
      if (contentResponse.ok) {
        const contentData = await contentResponse.json()
        setFeelGoodContent(contentData)
      }

    } catch (error) {
      console.error("Failed to fetch global data:", error)
      setError("Failed to load global statistics")
    } finally {
      setLoading(false)
    }
  }

  const getRiskLevelColor = (riskLevel: string) => {
    switch (riskLevel) {
      case "low": return "#10B981"
      case "moderate": return "#F59E0B"
      case "high": return "#EF4444"
      default: return "#6B7280"
    }
  }

  const getTrendIcon = (direction: string) => {
    switch (direction) {
      case "improving": return <TrendingUp className="w-5 h-5 text-green-600" />
      case "concerning": return <TrendingUp className="w-5 h-5 text-red-600 rotate-180" />
      default: return <Activity className="w-5 h-5 text-blue-600" />
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <Header language={language} />
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600">Loading global statistics...</p>
            </div>
          </div>
        </div>
        <Footer language={language} />
      </div>
    )
  }

  if (error || !globalStats) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <Header language={language} />
        <div className="container mx-auto px-4 py-8">
          <Alert className="max-w-md mx-auto border-red-200 bg-red-50">
            <AlertDescription className="text-red-800">
              {error || "Failed to load global statistics"}
            </AlertDescription>
          </Alert>
        </div>
        <Footer language={language} />
      </div>
    )
  }

  const riskDistributionData = Object.entries(globalStats.riskLevelDistribution).map(([key, value]) => ({
    name: key.charAt(0).toUpperCase() + key.slice(1),
    value,
    color: getRiskLevelColor(key)
  }))

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Header language={language} />
      
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Community Mental Health Insights
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Discover how our community is working together to promote mental wellness and support each other's journey.
          </p>
        </div>

        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="trends">Trends</TabsTrigger>
            <TabsTrigger value="insights">Insights</TabsTrigger>
            <TabsTrigger value="community">Community</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            {/* Key Statistics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Users</CardTitle>
                  <Users className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{globalStats.totalUsers.toLocaleString()}</div>
                  <p className="text-xs text-muted-foreground">
                    {globalStats.totalRegisteredUsers} registered, {globalStats.totalTemporaryUsers} guests
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Assessments</CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{globalStats.totalAssessments.toLocaleString()}</div>
                  <p className="text-xs text-muted-foreground">
                    Mental health check-ins completed
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Community Wellness</CardTitle>
                  <Heart className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    <Badge 
                      variant="outline" 
                      style={{ 
                        backgroundColor: getRiskLevelColor(globalStats.averageRiskLevel) + '20',
                        borderColor: getRiskLevelColor(globalStats.averageRiskLevel),
                        color: getRiskLevelColor(globalStats.averageRiskLevel)
                      }}
                    >
                      {globalStats.averageRiskLevel.charAt(0).toUpperCase() + globalStats.averageRiskLevel.slice(1)}
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Average community risk level
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Trend Direction</CardTitle>
                  {getTrendIcon(globalStats.mentalHealthTrends.trend_direction)}
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold capitalize">
                    {globalStats.mentalHealthTrends.trend_direction}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Community mental health trend
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Risk Level Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Community Risk Level Distribution</CardTitle>
                <CardDescription>
                  How our community members are feeling based on recent assessments
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={riskDistributionData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {riskDistributionData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="trends" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Mental Health Trends</CardTitle>
                <CardDescription>
                  Community mental health patterns over the past 8 weeks
                </CardDescription>
              </CardHeader>
              <CardContent>
                {globalStats.mentalHealthTrends.weekly_data.length > 0 ? (
                  <div className="h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={globalStats.mentalHealthTrends.weekly_data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                          dataKey="date" 
                          tickFormatter={(value) => new Date(value).toLocaleDateString()}
                        />
                        <YAxis />
                        <Tooltip 
                          labelFormatter={(value) => `Week of ${new Date(value).toLocaleDateString()}`}
                          formatter={(value: number, name: string) => [
                            name === 'averageScore' ? `${value} avg score` : `${value} assessments`,
                            name === 'averageScore' ? 'Average PHQ-9 Score' : 'Assessment Count'
                          ]}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="averageScore" 
                          stroke="#8884d8" 
                          strokeWidth={2}
                          name="averageScore"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <p className="text-gray-500">Not enough data to show trends yet.</p>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Trend Insights</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {globalStats.mentalHealthTrends.insights.map((insight, index) => (
                    <div key={index} className="flex items-start space-x-3">
                      <Lightbulb className="w-5 h-5 text-yellow-500 mt-0.5 flex-shrink-0" />
                      <p className="text-gray-700">{insight}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="insights" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Heart className="w-5 h-5 text-red-500" />
                    <span>Encouraging Messages</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {globalStats.encouragingMessages.map((message, index) => (
                      <div key={index} className="p-3 bg-green-50 border-l-4 border-green-400 rounded">
                        <p className="text-green-800">{message}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Lightbulb className="w-5 h-5 text-yellow-500" />
                    <span>Psychological Insights</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {globalStats.psychologicalInsights.map((insight, index) => (
                      <div key={index} className="p-3 bg-blue-50 border-l-4 border-blue-400 rounded">
                        <p className="text-blue-800">{insight}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Globe className="w-5 h-5 text-purple-500" />
                  <span>Platform Insights</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {globalStats.platformInsights.map((insight, index) => (
                    <div key={index} className="p-3 bg-purple-50 border-l-4 border-purple-400 rounded">
                      <p className="text-purple-800">{insight}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="community" className="space-y-6">
            {feelGoodContent && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Star className="w-5 h-5 text-yellow-500" />
                      <span>Daily Affirmations</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {feelGoodContent.daily_affirmations.map((affirmation, index) => (
                        <div key={index} className="p-3 bg-yellow-50 border border-yellow-200 rounded">
                          <p className="text-yellow-800 font-medium">{affirmation}</p>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Lightbulb className="w-5 h-5 text-blue-500" />
                      <span>Mental Health Tips</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {feelGoodContent.mental_health_tips.map((tip, index) => (
                        <div key={index} className="p-3 bg-blue-50 border border-blue-200 rounded">
                          <p className="text-blue-800">{tip}</p>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <MessageCircle className="w-5 h-5 text-green-500" />
                      <span>Success Stories</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {feelGoodContent.success_stories.map((story, index) => (
                        <div key={index} className="p-3 bg-green-50 border border-green-200 rounded">
                          <p className="text-green-800">{story}</p>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Users className="w-5 h-5 text-purple-500" />
                      <span>Community Impact</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {feelGoodContent.community_impact.map((impact, index) => (
                        <div key={index} className="p-3 bg-purple-50 border border-purple-200 rounded">
                          <p className="text-purple-800">{impact}</p>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>

      <Footer language={language} />
    </div>
  )
}

