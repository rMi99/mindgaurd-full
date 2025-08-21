"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog"
import { Calendar, Download, Filter, Trash2, FileText, TrendingUp, BarChart3, ArrowLeft, Plus, Eye } from "lucide-react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts"

interface DetailedAssessment {
  id: string
  date: string
  phq9Score: number
  riskLevel: string
  sleepData?: any
  demographicData?: any
  responses?: any
  aiInsights?: any
  notes?: string
}

interface HistoryStats {
  totalAssessments: number
  averagePhq9Score: number
  riskLevelDistribution: Record<string, number>
  assessmentFrequency: Record<string, number>
  improvementTrend: string
  lastAssessmentDate: string
}

export default function HistoryPage() {
  const [assessments, setAssessments] = useState<DetailedAssessment[]>([])
  const [stats, setStats] = useState<HistoryStats | null>(null)
  const [insights, setInsights] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [selectedAssessment, setSelectedAssessment] = useState<DetailedAssessment | null>(null)
  const [noteText, setNoteText] = useState("")
  const [filters, setFilters] = useState({
    startDate: "",
    endDate: "",
    riskLevels: "",
    limit: "50"
  })

  const [userId] = useState(() => {
    if (typeof window !== "undefined") {
      let id = localStorage.getItem("mindguard_user_id")
      if (!id) {
        id = "user_" + Math.random().toString(36).substr(2, 9)
        localStorage.setItem("mindguard_user_id", id)
      }
      return id
    }
    return "user_" + Math.random().toString(36).substr(2, 9)
  })

  useEffect(() => {
    fetchHistoryData()
    fetchStats()
    fetchInsights()
  }, [userId])

  const fetchHistoryData = async () => {
    try {
      setLoading(true)
      const params = new URLSearchParams({ userId })
      
      if (filters.startDate) params.append('startDate', filters.startDate)
      if (filters.endDate) params.append('endDate', filters.endDate)
      if (filters.riskLevels) params.append('riskLevels', filters.riskLevels)
      if (filters.limit) params.append('limit', filters.limit)

      const accessToken = localStorage.getItem('access_token')

      const response = await fetch(`/api/history?${params.toString()}`, {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json'
        }
      })
      if (response.ok) {
        const result = await response.json()
        // Handle both wrapped and direct responses
        if (result.status === "success" && result.data) {
          // If it's an array, use it directly; if it's an object with assessments, extract that
          if (Array.isArray(result.data)) {
            setAssessments(result.data)
          } else if (result.data.assessments && Array.isArray(result.data.assessments)) {
            setAssessments(result.data.assessments)
          } else {
            setAssessments([])
          }
        } else if (Array.isArray(result)) {
          setAssessments(result)
        } else {
          setAssessments([])
        }
      } else {
        setAssessments([])
      }
    } catch (error) {
      console.error("Failed to fetch history:", error)
    } finally {
      setLoading(false)
    }
  }

  const fetchStats = async () => {
    try {
      const accessToken = localStorage.getItem('access_token')
      const response = await fetch(`/api/history?userId=${userId}&action=stats`, {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json'
        }
      })
      if (response.ok) {
        const result = await response.json()
        // Handle wrapped response
        if (result.status === "success" && result.data) {
          setStats(result.data)
        } else {
          setStats(result)
        }
      }
    } catch (error) {
      console.error("Failed to fetch stats:", error)
    }
  }

  const fetchInsights = async () => {
    try {
      const accessToken = localStorage.getItem('access_token')
      const response = await fetch(`/api/history?userId=${userId}&action=insights`, {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json'
        }
      })
      if (response.ok) {
        const result = await response.json()
        // Handle wrapped response
        if (result.status === "success" && result.data) {
          setInsights(result.data)
        } else {
          setInsights(result)
        }
      }
    } catch (error) {
      console.error("Failed to fetch insights:", error)
    }
  }

  const exportData = async () => {
    try {
      const response = await fetch(`/api/history?userId=${userId}&action=export`, {
        method: 'POST'
      })
      if (response.ok) {
        const data = await response.json()
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `mindguard-history-${new Date().toISOString().split('T')[0]}.json`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
      }
    } catch (error) {
      console.error("Failed to export data:", error)
    }
  }

  const deleteHistory = async () => {
    try {
      const response = await fetch(`/api/history?userId=${userId}&confirm=true`, {
        method: 'DELETE'
      })
      if (response.ok) {
        setAssessments([])
        setStats(null)
        setInsights(null)
        await fetchHistoryData()
        await fetchStats()
        await fetchInsights()
      }
    } catch (error) {
      console.error("Failed to delete history:", error)
    }
  }

  const addNote = async (assessmentId: string, note: string) => {
    try {
      const response = await fetch(`/api/history?userId=${userId}&action=note`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ assessmentId, note })
      })
      if (response.ok) {
        await fetchHistoryData()
        setNoteText("")
        setSelectedAssessment(null)
      }
    } catch (error) {
      console.error("Failed to add note:", error)
    }
  }

  const getRiskLevelColor = (riskLevel: string) => {
    switch (riskLevel) {
      case "low": return "bg-green-100 text-green-800"
      case "moderate": return "bg-yellow-100 text-yellow-800"
      case "high": return "bg-red-100 text-red-800"
      default: return "bg-gray-100 text-gray-800"
    }
  }

  const COLORS = ['#10b981', '#f59e0b', '#ef4444']

  if (loading && assessments.length === 0) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600">Loading your history...</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <Button variant="ghost" size="sm" asChild className="mr-4">
                <a href="/dashboard">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back to Dashboard
                </a>
              </Button>
              <div className="flex-shrink-0">
                <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                  <BarChart3 className="w-6 h-6 text-white" />
                </div>
              </div>
              <div className="ml-4">
                <h1 className="text-2xl font-bold text-gray-900">Assessment History</h1>
                <p className="text-sm text-gray-600">Detailed view of your mental health journey</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Button onClick={exportData} variant="outline" size="sm">
                <Download className="w-4 h-4 mr-2" />
                Export Data
              </Button>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button variant="destructive" size="sm">
                    <Trash2 className="w-4 h-4 mr-2" />
                    Delete History
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Delete Assessment History</AlertDialogTitle>
                    <AlertDialogDescription>
                      This action cannot be undone. This will permanently delete all your assessment history and data.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={deleteHistory}>Delete</AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        {assessments.length === 0 ? (
          <Card className="text-center py-12">
            <CardContent>
              <BarChart3 className="h-16 w-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-gray-900 mb-2">No Assessment History</h3>
              <p className="text-gray-600 mb-6">
                You haven't completed any assessments yet. Start your mental health journey today.
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
              <TabsTrigger value="detailed">Detailed History</TabsTrigger>
              <TabsTrigger value="analytics">Analytics</TabsTrigger>
              <TabsTrigger value="insights">AI Insights</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-6">
              {/* Statistics Cards */}
              {stats && (
                <div className="grid md:grid-cols-4 gap-6">
                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Total Assessments</CardTitle>
                      <FileText className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{stats.totalAssessments}</div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Average PHQ-9</CardTitle>
                      <BarChart3 className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{stats.averagePhq9Score}</div>
                      <p className="text-xs text-muted-foreground">out of 27</p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Improvement Trend</CardTitle>
                      <TrendingUp className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold capitalize">{stats.improvementTrend.replace('_', ' ')}</div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Last Assessment</CardTitle>
                      <Calendar className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-sm font-bold">
                        {new Date(stats.lastAssessmentDate).toLocaleDateString()}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}

              {/* Risk Level Distribution */}
              {stats && (
                <Card>
                  <CardHeader>
                    <CardTitle>Risk Level Distribution</CardTitle>
                    <CardDescription>Breakdown of your assessment risk levels</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={Object.entries(stats.riskLevelDistribution).map(([level, count]) => ({
                            name: level,
                            value: count
                          }))}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {Object.entries(stats.riskLevelDistribution).map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="detailed" className="space-y-6">
              {/* Filters */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Filter className="w-5 h-5" />
                    Filters
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid md:grid-cols-4 gap-4">
                    <div>
                      <Label htmlFor="startDate">Start Date</Label>
                      <Input
                        id="startDate"
                        type="date"
                        value={filters.startDate}
                        onChange={(e) => setFilters({...filters, startDate: e.target.value})}
                      />
                    </div>
                    <div>
                      <Label htmlFor="endDate">End Date</Label>
                      <Input
                        id="endDate"
                        type="date"
                        value={filters.endDate}
                        onChange={(e) => setFilters({...filters, endDate: e.target.value})}
                      />
                    </div>
                    <div>
                      <Label htmlFor="riskLevels">Risk Levels</Label>
                      <Select value={filters.riskLevels} onValueChange={(value) => setFilters({...filters, riskLevels: value})}>
                        <SelectTrigger>
                          <SelectValue placeholder="All levels" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="">All levels</SelectItem>
                          <SelectItem value="low">Low</SelectItem>
                          <SelectItem value="moderate">Moderate</SelectItem>
                          <SelectItem value="high">High</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex items-end">
                      <Button onClick={fetchHistoryData} className="w-full">
                        Apply Filters
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Detailed Assessment List */}
              <Card>
                <CardHeader>
                  <CardTitle>Assessment Details</CardTitle>
                  <CardDescription>Complete record with notes and insights</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {Array.isArray(assessments) && assessments.length > 0 ? (
                      assessments.map((assessment) => (
                        <div key={assessment.id} className="border rounded-lg p-4">
                          <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center space-x-4">
                              <Calendar className="h-5 w-5 text-gray-400" />
                              <div>
                                <p className="font-medium">{new Date(assessment.date).toLocaleDateString()}</p>
                                <p className="text-sm text-gray-600">
                                  PHQ-9: {assessment.phq9Score}/27
                                </p>
                              </div>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Badge className={getRiskLevelColor(assessment.riskLevel)}>
                              {assessment.riskLevel}
                            </Badge>
                            <Dialog>
                              <DialogTrigger asChild>
                                <Button variant="outline" size="sm">
                                  <Eye className="w-4 h-4 mr-2" />
                                  View Details
                                </Button>
                              </DialogTrigger>
                              <DialogContent className="max-w-2xl">
                                <DialogHeader>
                                  <DialogTitle>Assessment Details</DialogTitle>
                                  <DialogDescription>
                                    {new Date(assessment.date).toLocaleDateString()}
                                  </DialogDescription>
                                </DialogHeader>
                                <div className="space-y-4">
                                  <div>
                                    <h4 className="font-semibold">PHQ-9 Score</h4>
                                    <p>{assessment.phq9Score}/27 ({assessment.riskLevel} risk)</p>
                                  </div>
                                  {assessment.sleepData && (
                                    <div>
                                      <h4 className="font-semibold">Sleep Data</h4>
                                      <pre className="text-sm bg-gray-100 p-2 rounded">
                                        {JSON.stringify(assessment.sleepData, null, 2)}
                                      </pre>
                                    </div>
                                  )}
                                  {assessment.notes && (
                                    <div>
                                      <h4 className="font-semibold">Notes</h4>
                                      <p className="text-sm">{assessment.notes}</p>
                                    </div>
                                  )}
                                </div>
                              </DialogContent>
                            </Dialog>
                            <Dialog>
                              <DialogTrigger asChild>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => {
                                    setSelectedAssessment(assessment)
                                    setNoteText(assessment.notes || "")
                                  }}
                                >
                                  <Plus className="w-4 h-4 mr-2" />
                                  Add Note
                                </Button>
                              </DialogTrigger>
                              <DialogContent>
                                <DialogHeader>
                                  <DialogTitle>Add Note</DialogTitle>
                                  <DialogDescription>
                                    Add a personal note to this assessment
                                  </DialogDescription>
                                </DialogHeader>
                                <div className="space-y-4">
                                  <Textarea
                                    placeholder="Enter your note here..."
                                    value={noteText}
                                    onChange={(e) => setNoteText(e.target.value)}
                                  />
                                  <Button
                                    onClick={() => selectedAssessment && addNote(selectedAssessment.id, noteText)}
                                  >
                                    Save Note
                                  </Button>
                                </div>
                              </DialogContent>
                            </Dialog>
                          </div>
                        </div>
                      </div>
                    ))
                    ) : (
                      <div className="text-center py-8">
                        <Calendar className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                        <h3 className="text-lg font-semibold text-gray-900 mb-2">No Assessments Found</h3>
                        <p className="text-gray-600">
                          No assessment history available. Take your first assessment to get started.
                        </p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="analytics" className="space-y-6">
              {/* PHQ-9 Trend Chart */}
              <Card>
                <CardHeader>
                  <CardTitle>PHQ-9 Score Trend</CardTitle>
                  <CardDescription>Your mental health scores over time</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={assessments.slice().reverse()}>
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
                        strokeWidth={3}
                        dot={{ fill: "#3b82f6", strokeWidth: 2, r: 6 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="insights" className="space-y-6">
              {insights && (
                <>
                  <Card>
                    <CardHeader>
                      <CardTitle>AI-Generated Insights</CardTitle>
                      <CardDescription>
                        Based on {insights.assessments_analyzed} assessments
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {insights.insights.map((insight: string, index: number) => (
                        <div key={index} className="p-4 bg-blue-50 rounded-lg">
                          <p className="text-sm text-blue-800">{insight}</p>
                        </div>
                      ))}
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Personalized Recommendations</CardTitle>
                      <CardDescription>Suggestions based on your patterns</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {insights.recommendations.map((recommendation: string, index: number) => (
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
    </div>
  )
}

