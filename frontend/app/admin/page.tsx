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
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Users, BarChart3, Shield, Flag, Trash2, Eye, Key, TrendingUp, AlertTriangle, Activity } from "lucide-react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar } from "recharts"

interface SystemStats {
  total_users: number
  total_assessments: number
  active_users_last_30_days: number
  average_assessments_per_user: number
  risk_level_distribution: Record<string, number>
  daily_assessment_counts: Record<string, number>
}

interface UserSummary {
  user_id: string
  total_assessments: number
  last_assessment_date: string
  average_phq9_score: number
  current_risk_level: string
  registration_date: string
  is_active: boolean
}

interface UserDetail {
  user_id: string
  assessments: Array<{
    id: string
    date: string
    phq9_score: number
    risk_level: string
    sleep_data: any
    notes: string
  }>
  statistics: any
  last_activity: string
  account_status: string
}

export default function AdminPage() {
  const [adminToken, setAdminToken] = useState("")
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null)
  const [users, setUsers] = useState<UserSummary[]>([])
  const [selectedUser, setSelectedUser] = useState<UserDetail | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const authenticate = async () => {
    if (adminToken === "mindguard_admin_2024") {
      setIsAuthenticated(true)
      await fetchSystemStats()
      await fetchUsers()
    } else {
      setError("Invalid admin token")
    }
  }

  const fetchSystemStats = async () => {
    try {
      const response = await fetch(`/api/admin/stats?token=${adminToken}`)
      if (response.ok) {
        const data = await response.json()
        setSystemStats(data)
      }
    } catch (error) {
      console.error("Failed to fetch system stats:", error)
    }
  }

  const fetchUsers = async () => {
    try {
      setLoading(true)
      const response = await fetch(`/api/admin/users?token=${adminToken}&limit=100`)
      if (response.ok) {
        const data = await response.json()
        setUsers(data)
      }
    } catch (error) {
      console.error("Failed to fetch users:", error)
    } finally {
      setLoading(false)
    }
  }

  const fetchUserDetail = async (userId: string) => {
    try {
      const response = await fetch(`/api/admin/users/${userId}?token=${adminToken}`)
      if (response.ok) {
        const data = await response.json()
        setSelectedUser(data)
      }
    } catch (error) {
      console.error("Failed to fetch user detail:", error)
    }
  }

  const deleteUser = async (userId: string, reason: string) => {
    try {
      const response = await fetch(`/api/admin/users/${userId}?token=${adminToken}&reason=${encodeURIComponent(reason)}`, {
        method: 'DELETE'
      })
      if (response.ok) {
        await fetchUsers()
        setSelectedUser(null)
      }
    } catch (error) {
      console.error("Failed to delete user:", error)
    }
  }

  const flagUser = async (userId: string, reason: string, priority: string) => {
    try {
      const response = await fetch(`/api/admin/users/${userId}/flag?token=${adminToken}&reason=${encodeURIComponent(reason)}&priority=${priority}`, {
        method: 'POST'
      })
      if (response.ok) {
        await fetchUsers()
      }
    } catch (error) {
      console.error("Failed to flag user:", error)
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

  const COLORS = ['#10b981', '#f59e0b', '#ef4444', '#6b7280']

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="w-6 h-6" />
              Admin Authentication
            </CardTitle>
            <CardDescription>Enter admin token to access the management panel</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="token">Admin Token</Label>
              <Input
                id="token"
                type="password"
                value={adminToken}
                onChange={(e) => setAdminToken(e.target.value)}
                placeholder="Enter admin token"
              />
            </div>
            {error && (
              <div className="text-red-600 text-sm">{error}</div>
            )}
            <Button onClick={authenticate} className="w-full">
              Authenticate
            </Button>
          </CardContent>
        </Card>
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
              <div className="flex-shrink-0">
                <div className="w-10 h-10 bg-red-600 rounded-lg flex items-center justify-center">
                  <Shield className="w-6 h-6 text-white" />
                </div>
              </div>
              <div className="ml-4">
                <h1 className="text-2xl font-bold text-gray-900">Admin Panel</h1>
                <p className="text-sm text-gray-600">MindGuard System Management</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant="destructive">Admin Access</Badge>
              <Button variant="outline" onClick={() => setIsAuthenticated(false)}>
                Logout
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="users">User Management</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
            <TabsTrigger value="flags">Flagged Users</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            {/* System Statistics */}
            {systemStats && (
              <>
                <div className="grid md:grid-cols-4 gap-6">
                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Total Users</CardTitle>
                      <Users className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{systemStats.total_users}</div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Total Assessments</CardTitle>
                      <BarChart3 className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{systemStats.total_assessments}</div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Active Users (30d)</CardTitle>
                      <Activity className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{systemStats.active_users_last_30_days}</div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Avg Assessments/User</CardTitle>
                      <TrendingUp className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{systemStats.average_assessments_per_user}</div>
                    </CardContent>
                  </Card>
                </div>

                {/* Risk Level Distribution */}
                <Card>
                  <CardHeader>
                    <CardTitle>Risk Level Distribution</CardTitle>
                    <CardDescription>Current distribution of user risk levels</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={Object.entries(systemStats.risk_level_distribution).map(([level, count]) => ({
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
                          {Object.entries(systemStats.risk_level_distribution).map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                {/* Daily Assessment Activity */}
                <Card>
                  <CardHeader>
                    <CardTitle>Daily Assessment Activity</CardTitle>
                    <CardDescription>Assessment counts over the last 30 days</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={Object.entries(systemStats.daily_assessment_counts).map(([date, count]) => ({
                        date,
                        count
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" tickFormatter={(value) => new Date(value).toLocaleDateString()} />
                        <YAxis />
                        <Tooltip labelFormatter={(value) => new Date(value).toLocaleDateString()} />
                        <Bar dataKey="count" fill="#3b82f6" />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </>
            )}
          </TabsContent>

          <TabsContent value="users" className="space-y-6">
            {/* User Management */}
            <Card>
              <CardHeader>
                <CardTitle>User Management</CardTitle>
                <CardDescription>Manage user accounts and view detailed information</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="text-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                    <p>Loading users...</p>
                  </div>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>User ID</TableHead>
                        <TableHead>Assessments</TableHead>
                        <TableHead>Last Activity</TableHead>
                        <TableHead>Risk Level</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {users.map((user) => (
                        <TableRow key={user.user_id}>
                          <TableCell className="font-mono text-sm">{user.user_id}</TableCell>
                          <TableCell>{user.total_assessments}</TableCell>
                          <TableCell>
                            {user.last_assessment_date ? new Date(user.last_assessment_date).toLocaleDateString() : 'Never'}
                          </TableCell>
                          <TableCell>
                            <Badge className={getRiskLevelColor(user.current_risk_level)}>
                              {user.current_risk_level}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <Badge variant={user.is_active ? "default" : "secondary"}>
                              {user.is_active ? "Active" : "Inactive"}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <div className="flex space-x-2">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => fetchUserDetail(user.user_id)}
                              >
                                <Eye className="w-4 h-4" />
                              </Button>
                              <Dialog>
                                <DialogTrigger asChild>
                                  <Button variant="outline" size="sm">
                                    <Flag className="w-4 h-4" />
                                  </Button>
                                </DialogTrigger>
                                <DialogContent>
                                  <DialogHeader>
                                    <DialogTitle>Flag User</DialogTitle>
                                    <DialogDescription>Flag this user for manual review</DialogDescription>
                                  </DialogHeader>
                                  <div className="space-y-4">
                                    <div>
                                      <Label>Reason</Label>
                                      <Textarea placeholder="Enter reason for flagging..." />
                                    </div>
                                    <div>
                                      <Label>Priority</Label>
                                      <Select defaultValue="medium">
                                        <SelectTrigger>
                                          <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                          <SelectItem value="low">Low</SelectItem>
                                          <SelectItem value="medium">Medium</SelectItem>
                                          <SelectItem value="high">High</SelectItem>
                                        </SelectContent>
                                      </Select>
                                    </div>
                                    <Button onClick={() => flagUser(user.user_id, "Manual review", "medium")}>
                                      Flag User
                                    </Button>
                                  </div>
                                </DialogContent>
                              </Dialog>
                              <AlertDialog>
                                <AlertDialogTrigger asChild>
                                  <Button variant="destructive" size="sm">
                                    <Trash2 className="w-4 h-4" />
                                  </Button>
                                </AlertDialogTrigger>
                                <AlertDialogContent>
                                  <AlertDialogHeader>
                                    <AlertDialogTitle>Delete User Account</AlertDialogTitle>
                                    <AlertDialogDescription>
                                      This will permanently delete the user account and all associated data.
                                    </AlertDialogDescription>
                                  </AlertDialogHeader>
                                  <AlertDialogFooter>
                                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                                    <AlertDialogAction onClick={() => deleteUser(user.user_id, "Admin deletion")}>
                                      Delete
                                    </AlertDialogAction>
                                  </AlertDialogFooter>
                                </AlertDialogContent>
                              </AlertDialog>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>

            {/* User Detail Modal */}
            {selectedUser && (
              <Card>
                <CardHeader>
                  <CardTitle>User Details: {selectedUser.user_id}</CardTitle>
                  <CardDescription>Complete user information and assessment history</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-semibold mb-2">Account Information</h4>
                      <div className="space-y-2 text-sm">
                        <p><strong>User ID:</strong> {selectedUser.user_id}</p>
                        <p><strong>Total Assessments:</strong> {selectedUser.statistics.total_assessments}</p>
                        <p><strong>Average PHQ-9:</strong> {selectedUser.statistics.average_phq9_score}</p>
                        <p><strong>Account Status:</strong> {selectedUser.account_status}</p>
                        <p><strong>Last Activity:</strong> {selectedUser.last_activity ? new Date(selectedUser.last_activity).toLocaleDateString() : 'Never'}</p>
                      </div>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-2">Recent Assessments</h4>
                      <div className="space-y-2">
                        {selectedUser.assessments.slice(0, 5).map((assessment, index) => (
                          <div key={index} className="text-sm border rounded p-2">
                            <p><strong>Date:</strong> {new Date(assessment.date).toLocaleDateString()}</p>
                            <p><strong>PHQ-9:</strong> {assessment.phq9_score}/27</p>
                            <p><strong>Risk:</strong> {assessment.risk_level}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="analytics" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>System Analytics</CardTitle>
                <CardDescription>Advanced analytics and reporting</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600">Advanced analytics features would be implemented here, including:</p>
                <ul className="list-disc list-inside mt-4 space-y-2 text-sm text-gray-600">
                  <li>Risk trend analysis over time</li>
                  <li>User engagement patterns</li>
                  <li>Assessment completion rates</li>
                  <li>Geographic distribution (if available)</li>
                  <li>Intervention effectiveness tracking</li>
                </ul>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="flags" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Flagged Users</CardTitle>
                <CardDescription>Users flagged for manual review</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600">Flagged users management interface would be implemented here.</p>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

