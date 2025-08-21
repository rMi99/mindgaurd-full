"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Calendar, User, Mail, Phone, MapPin, Briefcase, Edit, Save, X, Shield, Clock, Activity } from "lucide-react"
import { useRouter } from "next/navigation"
import Header from "../components/Header"
import Footer from "../components/Footer"
import type { Language } from "@/lib/types"

interface UserProfile {
  id: string
  email: string
  full_name: string
  username?: string
  age?: number
  gender?: string
  phone?: string
  location?: string
  occupation?: string
  bio?: string
  created_at: string
  last_login?: string
  is_authenticated: boolean
  account_status: string
  profile_picture?: string
}

export default function ProfilePage() {
  const router = useRouter()
  const [language, setLanguage] = useState<Language>("en")
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [editingProfile, setEditingProfile] = useState<UserProfile | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isEditing, setIsEditing] = useState(false)
  const [success, setSuccess] = useState<string | null>(null)

  useEffect(() => {
    fetchProfile()
  }, [])

  const fetchProfile = async () => {
    try {
      setLoading(true)
      setError(null)

      const token = localStorage.getItem('access_token')
      if (!token) {
        router.push('/auth')
        return
      }

      const response = await fetch('/api/profile', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        if (response.status === 401) {
          localStorage.clear()
          router.push('/auth')
          return
        }
        throw new Error('Failed to fetch profile')
      }

      const data = await response.json()
      setProfile(data)
      setEditingProfile(data)
    } catch (error) {
      console.error('Profile fetch error:', error)
      setError('Failed to load profile data')
    } finally {
      setLoading(false)
    }
  }

  const handleEdit = () => {
    setIsEditing(true)
    setEditingProfile({ ...profile! })
    setError(null)
    setSuccess(null)
  }

  const handleCancel = () => {
    setIsEditing(false)
    setEditingProfile({ ...profile! })
    setError(null)
    setSuccess(null)
  }

  const handleSave = async () => {
    try {
      setSaving(true)
      setError(null)

      const token = localStorage.getItem('access_token')
      const response = await fetch('/api/profile', {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(editingProfile),
      })

      if (!response.ok) {
        throw new Error('Failed to update profile')
      }

      const updatedProfile = await response.json()
      setProfile(updatedProfile)
      setIsEditing(false)
      setSuccess('Profile updated successfully!')
      
      // Clear success message after 3 seconds
      setTimeout(() => setSuccess(null), 3000)
    } catch (error) {
      console.error('Profile update error:', error)
      setError('Failed to update profile')
    } finally {
      setSaving(false)
    }
  }

  const handleInputChange = (field: keyof UserProfile, value: string | number) => {
    if (editingProfile) {
      setEditingProfile({
        ...editingProfile,
        [field]: value,
      })
    }
  }

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map(n => n[0])
      .join('')
      .toUpperCase()
      .slice(0, 2)
  }

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col">
        <Header language={language} />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-lg text-gray-600">Loading your profile...</p>
          </div>
        </main>
        <Footer language={language} />
      </div>
    )
  }

  if (!profile) {
    return (
      <div className="min-h-screen flex flex-col">
        <Header language={language} />
        <main className="flex-1 flex items-center justify-center">
          <Card className="w-full max-w-md">
            <CardContent className="pt-6">
              <Alert>
                <AlertDescription>
                  {error || "Unable to load profile data"}
                </AlertDescription>
              </Alert>
              <div className="mt-4 space-x-2">
                <Button onClick={fetchProfile}>Retry</Button>
                <Button variant="outline" onClick={() => router.push('/dashboard')}>
                  Back to Dashboard
                </Button>
              </div>
            </CardContent>
          </Card>
        </main>
        <Footer language={language} />
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header language={language} />

      <main className="flex-1 px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Success Message */}
          {success && (
            <Alert className="mb-6 border-green-200 bg-green-50">
              <AlertDescription className="text-green-800">
                {success}
              </AlertDescription>
            </Alert>
          )}

          {/* Error Message */}
          {error && (
            <Alert className="mb-6 border-red-200 bg-red-50">
              <AlertDescription className="text-red-800">
                {error}
              </AlertDescription>
            </Alert>
          )}

          {/* Profile Header */}
          <Card className="mb-8">
            <CardContent className="pt-6">
              <div className="flex items-center space-x-6">
                <Avatar className="h-24 w-24">
                  <AvatarImage src={profile.profile_picture} />
                  <AvatarFallback className="text-lg">
                    {getInitials(profile.full_name || profile.email)}
                  </AvatarFallback>
                </Avatar>
                <div className="flex-1">
                  <h1 className="text-3xl font-bold text-gray-900">{profile.full_name || profile.email}</h1>
                  <p className="text-gray-600">{profile.email}</p>
                  {profile.occupation && (
                    <p className="text-gray-500">{profile.occupation}</p>
                  )}
                  <div className="flex items-center space-x-4 mt-2">
                    <Badge variant={profile.is_authenticated ? "default" : "secondary"}>
                      {profile.is_authenticated ? "Verified Account" : "Guest User"}
                    </Badge>
                    <Badge variant="outline">{profile.account_status}</Badge>
                  </div>
                </div>
                <div className="space-x-2">
                  {!isEditing ? (
                    <Button onClick={handleEdit}>
                      <Edit className="h-4 w-4 mr-2" />
                      Edit Profile
                    </Button>
                  ) : (
                    <div className="space-x-2">
                      <Button onClick={handleSave} disabled={saving}>
                        <Save className="h-4 w-4 mr-2" />
                        {saving ? 'Saving...' : 'Save'}
                      </Button>
                      <Button variant="outline" onClick={handleCancel}>
                        <X className="h-4 w-4 mr-2" />
                        Cancel
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          <Tabs defaultValue="personal" className="space-y-6">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="personal">Personal Information</TabsTrigger>
              <TabsTrigger value="account">Account Details</TabsTrigger>
              <TabsTrigger value="activity">Activity</TabsTrigger>
            </TabsList>

            <TabsContent value="personal" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Personal Information</CardTitle>
                  <CardDescription>
                    Update your personal details and bio
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <Label htmlFor="full_name">Full Name</Label>
                      {isEditing ? (
                        <Input
                          id="full_name"
                          value={editingProfile?.full_name || ''}
                          onChange={(e) => handleInputChange('full_name', e.target.value)}
                          placeholder="Enter your full name"
                        />
                      ) : (
                        <p className="text-gray-900">{profile.full_name || 'Not provided'}</p>
                      )}
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="username">Username</Label>
                      {isEditing ? (
                        <Input
                          id="username"
                          value={editingProfile?.username || ''}
                          onChange={(e) => handleInputChange('username', e.target.value)}
                          placeholder="Enter your username"
                        />
                      ) : (
                        <p className="text-gray-900">{profile.username || 'Not provided'}</p>
                      )}
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="age">Age</Label>
                      {isEditing ? (
                        <Input
                          id="age"
                          type="number"
                          value={editingProfile?.age || ''}
                          onChange={(e) => handleInputChange('age', parseInt(e.target.value) || 0)}
                          placeholder="Enter your age"
                        />
                      ) : (
                        <p className="text-gray-900">{profile.age || 'Not provided'}</p>
                      )}
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="gender">Gender</Label>
                      {isEditing ? (
                        <Input
                          id="gender"
                          value={editingProfile?.gender || ''}
                          onChange={(e) => handleInputChange('gender', e.target.value)}
                          placeholder="Enter your gender"
                        />
                      ) : (
                        <p className="text-gray-900">{profile.gender || 'Not provided'}</p>
                      )}
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="phone">Phone</Label>
                      {isEditing ? (
                        <Input
                          id="phone"
                          value={editingProfile?.phone || ''}
                          onChange={(e) => handleInputChange('phone', e.target.value)}
                          placeholder="Enter your phone number"
                        />
                      ) : (
                        <p className="text-gray-900">{profile.phone || 'Not provided'}</p>
                      )}
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="location">Location</Label>
                      {isEditing ? (
                        <Input
                          id="location"
                          value={editingProfile?.location || ''}
                          onChange={(e) => handleInputChange('location', e.target.value)}
                          placeholder="Enter your location"
                        />
                      ) : (
                        <p className="text-gray-900">{profile.location || 'Not provided'}</p>
                      )}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="occupation">Occupation</Label>
                    {isEditing ? (
                      <Input
                        id="occupation"
                        value={editingProfile?.occupation || ''}
                        onChange={(e) => handleInputChange('occupation', e.target.value)}
                        placeholder="Enter your occupation"
                      />
                    ) : (
                      <p className="text-gray-900">{profile.occupation || 'Not provided'}</p>
                    )}
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="bio">Bio</Label>
                    {isEditing ? (
                      <Textarea
                        id="bio"
                        value={editingProfile?.bio || ''}
                        onChange={(e) => handleInputChange('bio', e.target.value)}
                        placeholder="Tell us about yourself"
                        rows={4}
                      />
                    ) : (
                      <p className="text-gray-900">{profile.bio || 'Not provided'}</p>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="account" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Account Details</CardTitle>
                  <CardDescription>
                    View your account information and security details
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="flex items-center space-x-3">
                      <Mail className="h-5 w-5 text-gray-400" />
                      <div>
                        <p className="text-sm font-medium text-gray-500">Email</p>
                        <p className="text-gray-900">{profile.email}</p>
                      </div>
                    </div>

                    <div className="flex items-center space-x-3">
                      <User className="h-5 w-5 text-gray-400" />
                      <div>
                        <p className="text-sm font-medium text-gray-500">User ID</p>
                        <p className="text-gray-900 font-mono text-sm">{profile.id}</p>
                      </div>
                    </div>

                    <div className="flex items-center space-x-3">
                      <Calendar className="h-5 w-5 text-gray-400" />
                      <div>
                        <p className="text-sm font-medium text-gray-500">Member Since</p>
                        <p className="text-gray-900">
                          {new Date(profile.created_at).toLocaleDateString()}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center space-x-3">
                      <Clock className="h-5 w-5 text-gray-400" />
                      <div>
                        <p className="text-sm font-medium text-gray-500">Last Login</p>
                        <p className="text-gray-900">
                          {profile.last_login 
                            ? new Date(profile.last_login).toLocaleDateString()
                            : 'Never'
                          }
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center space-x-3">
                      <Shield className="h-5 w-5 text-gray-400" />
                      <div>
                        <p className="text-sm font-medium text-gray-500">Account Status</p>
                        <Badge>{profile.account_status}</Badge>
                      </div>
                    </div>
                  </div>

                  <div className="pt-6 border-t">
                    <h4 className="text-lg font-medium mb-4">Security Actions</h4>
                    <div className="space-x-4">
                      <Button variant="outline">Change Password</Button>
                      <Button variant="outline">Enable Two-Factor Auth</Button>
                      <Button variant="outline">Download Data</Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="activity" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Activity Overview</CardTitle>
                  <CardDescription>
                    Your recent activity and usage statistics
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-center h-32">
                    <div className="text-center text-gray-500">
                      <Activity className="h-8 w-8 mx-auto mb-2" />
                      <p>Activity tracking coming soon</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </main>

      <Footer language={language} />
    </div>
  )
}
