"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"
import { Trash2, Settings, User, Mail, Lock, AlertTriangle, CheckCircle } from "lucide-react"
import { useRouter } from "next/navigation"
import Header from "../components/Header"
import Footer from "../components/Footer"
import type { Language } from "@/lib/types"

interface UserProfile {
  user_id: string
  email: string
  username?: string
  created_at: string
  last_login?: string
  auth_provider: string
  account_status: string
}

interface Assessment {
  id: string
  timestamp: string
  phq9_score: number
  risk_level: string
  sleep_data: any
}

export default function UserManagementPage() {
  const router = useRouter()
  const [language, setLanguage] = useState<Language>("en")
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null)
  const [assessments, setAssessments] = useState<Assessment[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  // Form states
  const [currentPassword, setCurrentPassword] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [newEmail, setNewEmail] = useState("")
  const [emailPassword, setEmailPassword] = useState("")
  const [newUsername, setNewUsername] = useState("")

  // Dialog states
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [showDeleteHistoryConfirm, setShowDeleteHistoryConfirm] = useState(false)
  const [selectedAssessments, setSelectedAssessments] = useState<string[]>([])

  useEffect(() => {
    checkAuthAndFetchData()
  }, [])

  const checkAuthAndFetchData = async () => {
    try {
      setLoading(true)
      setError(null)

      const accessToken = localStorage.getItem('access_token')
      if (!accessToken) {
        router.push('/auth')
        return
      }

      await Promise.all([fetchUserProfile(), fetchAssessmentHistory()])
    } catch (error) {
      console.error("Failed to fetch data:", error)
      setError("Failed to load user data")
    } finally {
      setLoading(false)
    }
  }

  const fetchUserProfile = async () => {
    const accessToken = localStorage.getItem('access_token')
    const response = await fetch('/api/user/profile', {
      headers: {
        'Authorization': `Bearer ${accessToken}`,
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      if (response.status === 401) {
        localStorage.removeItem('access_token')
        router.push('/auth')
        return
      }
      throw new Error('Failed to fetch user profile')
    }

    const data = await response.json()
    setUserProfile(data)
    setNewEmail(data.email)
    setNewUsername(data.username || "")
  }

  const fetchAssessmentHistory = async () => {
    const accessToken = localStorage.getItem('access_token')
    const response = await fetch('/api/user/assessment-history', {
      headers: {
        'Authorization': `Bearer ${accessToken}`,
        'Content-Type': 'application/json',
      },
    })

    if (response.ok) {
      const data = await response.json()
      setAssessments(data.assessments || [])
    }
  }

  const handlePasswordChange = async () => {
    if (newPassword !== confirmPassword) {
      setError("New passwords don't match")
      return
    }

    if (newPassword.length < 6) {
      setError("Password must be at least 6 characters long")
      return
    }

    try {
      const accessToken = localStorage.getItem('access_token')
      const response = await fetch('/api/user/change-password', {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          current_password: currentPassword,
          new_password: newPassword
        })
      })

      const data = await response.json()

      if (response.ok) {
        setSuccess("Password changed successfully")
        setCurrentPassword("")
        setNewPassword("")
        setConfirmPassword("")
      } else {
        setError(data.detail || "Failed to change password")
      }
    } catch (error) {
      setError("Failed to change password")
    }
  }

  const handleEmailChange = async () => {
    try {
      const accessToken = localStorage.getItem('access_token')
      const body: any = { new_email: newEmail }
      
      if (userProfile?.auth_provider === 'email') {
        body.password = emailPassword
      }

      const response = await fetch('/api/user/change-email', {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body)
      })

      const data = await response.json()

      if (response.ok) {
        setSuccess("Email changed successfully")
        setEmailPassword("")
        await fetchUserProfile()
      } else {
        setError(data.detail || "Failed to change email")
      }
    } catch (error) {
      setError("Failed to change email")
    }
  }

  const handleUsernameChange = async () => {
    try {
      const accessToken = localStorage.getItem('access_token')
      const response = await fetch('/api/user/change-username', {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ new_username: newUsername })
      })

      const data = await response.json()

      if (response.ok) {
        setSuccess("Username changed successfully")
        await fetchUserProfile()
      } else {
        setError(data.detail || "Failed to change username")
      }
    } catch (error) {
      setError("Failed to change username")
    }
  }

  const handleDeleteAssessment = async (assessmentId: string) => {
    try {
      const accessToken = localStorage.getItem('access_token')
      const response = await fetch(`/api/user/assessment/${assessmentId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
        },
      })

      if (response.ok) {
        setSuccess("Assessment deleted successfully")
        await fetchAssessmentHistory()
      } else {
        setError("Failed to delete assessment")
      }
    } catch (error) {
      setError("Failed to delete assessment")
    }
  }

  const handleDeleteAllHistory = async () => {
    try {
      const accessToken = localStorage.getItem('access_token')
      const response = await fetch('/api/user/delete-history', {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({})
      })

      const data = await response.json()

      if (response.ok) {
        setSuccess("All assessment history deleted successfully")
        setShowDeleteHistoryConfirm(false)
        await fetchAssessmentHistory()
      } else {
        setError(data.detail || "Failed to delete history")
      }
    } catch (error) {
      setError("Failed to delete history")
    }
  }

  const handleDeleteAccount = async () => {
    try {
      const accessToken = localStorage.getItem('access_token')
      const response = await fetch('/api/user/delete-account', {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
        },
      })

      if (response.ok) {
        localStorage.clear()
        router.push('/')
      } else {
        setError("Failed to delete account")
      }
    } catch (error) {
      setError("Failed to delete account")
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

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <Header language={language} />
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600">Loading user management...</p>
            </div>
          </div>
        </div>
        <Footer language={language} />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Header language={language} />
      
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Account Management
          </h1>
          <p className="text-xl text-gray-600">
            Manage your account settings and data
          </p>
        </div>

        {error && (
          <Alert className="mb-6 border-red-200 bg-red-50">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription className="text-red-800">{error}</AlertDescription>
          </Alert>
        )}

        {success && (
          <Alert className="mb-6 border-green-200 bg-green-50">
            <CheckCircle className="h-4 w-4" />
            <AlertDescription className="text-green-800">{success}</AlertDescription>
          </Alert>
        )}

        <Tabs defaultValue="profile" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="profile">Profile</TabsTrigger>
            <TabsTrigger value="security">Security</TabsTrigger>
            <TabsTrigger value="history">History</TabsTrigger>
            <TabsTrigger value="danger">Danger Zone</TabsTrigger>
          </TabsList>

          <TabsContent value="profile" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <User className="w-5 h-5" />
                  <span>Profile Information</span>
                </CardTitle>
                <CardDescription>
                  Update your personal information
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {userProfile && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <Label htmlFor="email">Email</Label>
                      <div className="flex space-x-2">
                        <Input
                          id="email"
                          type="email"
                          value={newEmail}
                          onChange={(e) => setNewEmail(e.target.value)}
                        />
                        <Button onClick={handleEmailChange}>Update</Button>
                      </div>
                      {userProfile.auth_provider === 'email' && (
                        <div className="mt-2">
                          <Input
                            type="password"
                            placeholder="Current password"
                            value={emailPassword}
                            onChange={(e) => setEmailPassword(e.target.value)}
                          />
                        </div>
                      )}
                    </div>

                    <div>
                      <Label htmlFor="username">Username</Label>
                      <div className="flex space-x-2">
                        <Input
                          id="username"
                          value={newUsername}
                          onChange={(e) => setNewUsername(e.target.value)}
                        />
                        <Button onClick={handleUsernameChange}>Update</Button>
                      </div>
                    </div>

                    <div>
                      <Label>Account Type</Label>
                      <Badge variant="outline" className="ml-2">
                        {userProfile.auth_provider === 'google' ? 'Google Account' : 'Email Account'}
                      </Badge>
                    </div>

                    <div>
                      <Label>Member Since</Label>
                      <p className="text-sm text-gray-600">
                        {new Date(userProfile.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="security" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Lock className="w-5 h-5" />
                  <span>Change Password</span>
                </CardTitle>
                <CardDescription>
                  {userProfile?.auth_provider === 'google' 
                    ? 'Password change is not available for Google accounts'
                    : 'Update your account password'
                  }
                </CardDescription>
              </CardHeader>
              <CardContent>
                {userProfile?.auth_provider === 'email' ? (
                  <div className="space-y-4 max-w-md">
                    <div>
                      <Label htmlFor="current-password">Current Password</Label>
                      <Input
                        id="current-password"
                        type="password"
                        value={currentPassword}
                        onChange={(e) => setCurrentPassword(e.target.value)}
                      />
                    </div>
                    <div>
                      <Label htmlFor="new-password">New Password</Label>
                      <Input
                        id="new-password"
                        type="password"
                        value={newPassword}
                        onChange={(e) => setNewPassword(e.target.value)}
                      />
                    </div>
                    <div>
                      <Label htmlFor="confirm-password">Confirm New Password</Label>
                      <Input
                        id="confirm-password"
                        type="password"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                      />
                    </div>
                    <Button onClick={handlePasswordChange}>Change Password</Button>
                  </div>
                ) : (
                  <p className="text-gray-600">
                    Your account is managed through Google. Please use Google's account settings to change your password.
                  </p>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="history" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Assessment History</CardTitle>
                <CardDescription>
                  View and manage your mental health assessment history
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <p className="text-sm text-gray-600">
                      Total assessments: {assessments.length}
                    </p>
                    <Dialog open={showDeleteHistoryConfirm} onOpenChange={setShowDeleteHistoryConfirm}>
                      <DialogTrigger asChild>
                        <Button variant="destructive" size="sm">
                          Delete All History
                        </Button>
                      </DialogTrigger>
                      <DialogContent>
                        <DialogHeader>
                          <DialogTitle>Delete All Assessment History</DialogTitle>
                          <DialogDescription>
                            This will permanently delete all your mental health assessments. This action cannot be undone.
                          </DialogDescription>
                        </DialogHeader>
                        <DialogFooter>
                          <Button variant="outline" onClick={() => setShowDeleteHistoryConfirm(false)}>
                            Cancel
                          </Button>
                          <Button variant="destructive" onClick={handleDeleteAllHistory}>
                            Delete All
                          </Button>
                        </DialogFooter>
                      </DialogContent>
                    </Dialog>
                  </div>

                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {assessments.map((assessment) => (
                      <div key={assessment.id} className="flex items-center justify-between p-3 border rounded">
                        <div className="flex items-center space-x-4">
                          <div>
                            <p className="text-sm font-medium">
                              {new Date(assessment.timestamp).toLocaleDateString()}
                            </p>
                            <p className="text-xs text-gray-600">
                              PHQ-9 Score: {assessment.phq9_score}
                            </p>
                          </div>
                          <Badge className={getRiskLevelColor(assessment.risk_level)}>
                            {assessment.risk_level}
                          </Badge>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDeleteAssessment(assessment.id)}
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="danger" className="space-y-6">
            <Card className="border-red-200">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2 text-red-600">
                  <AlertTriangle className="w-5 h-5" />
                  <span>Danger Zone</span>
                </CardTitle>
                <CardDescription>
                  Irreversible and destructive actions
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-4 border border-red-200 rounded bg-red-50">
                    <h3 className="font-medium text-red-800 mb-2">Delete Account</h3>
                    <p className="text-sm text-red-700 mb-4">
                      This will permanently delete your account and all associated data. This action cannot be undone.
                    </p>
                    <Dialog open={showDeleteConfirm} onOpenChange={setShowDeleteConfirm}>
                      <DialogTrigger asChild>
                        <Button variant="destructive">Delete Account</Button>
                      </DialogTrigger>
                      <DialogContent>
                        <DialogHeader>
                          <DialogTitle>Delete Account</DialogTitle>
                          <DialogDescription>
                            Are you absolutely sure? This will permanently delete your account and all your data. This action cannot be undone.
                          </DialogDescription>
                        </DialogHeader>
                        <DialogFooter>
                          <Button variant="outline" onClick={() => setShowDeleteConfirm(false)}>
                            Cancel
                          </Button>
                          <Button variant="destructive" onClick={handleDeleteAccount}>
                            Delete Account
                          </Button>
                        </DialogFooter>
                      </DialogContent>
                    </Dialog>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      <Footer language={language} />
    </div>
  )
}

