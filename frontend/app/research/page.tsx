"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ArrowLeft, BookOpen, Users, Target, Search } from "lucide-react"

interface ResearchSection {
  id: string
  title: string
  content: string
  subsections: Array<{
    title: string
    content: string
  }>
}

interface ResearchData {
  sections: ResearchSection[]
  metadata: {
    last_updated: string
    version: string
    contributors: string[]
    review_status: string
    citation_count: number
    keywords: string[]
  }
}

export default function ResearchPage() {
  const [researchData, setResearchData] = useState<ResearchData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchResearchData()
  }, [])

  const fetchResearchData = async () => {
    try {
      const response = await fetch('/api/research')
      if (!response.ok) {
        throw new Error('Failed to fetch research data')
      }
      const data = await response.json()
      setResearchData(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const getSectionIcon = (sectionId: string) => {
    switch (sectionId) {
      case 'methodology':
        return <Search className="w-5 h-5" />
      case 'objectives':
        return <Target className="w-5 h-5" />
      case 'background':
        return <BookOpen className="w-5 h-5" />
      case 'literature_review':
        return <Users className="w-5 h-5" />
      default:
        return <BookOpen className="w-5 h-5" />
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600">Loading research content...</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center">
              <p className="text-red-600 mb-4">Error: {error}</p>
              <Button onClick={fetchResearchData}>Try Again</Button>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (!researchData) {
    return null
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <Button variant="ghost" size="sm" asChild className="mr-4">
                <a href="/">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back to Home
                </a>
              </Button>
              <div className="flex-shrink-0">
                <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                  <BookOpen className="w-6 h-6 text-white" />
                </div>
              </div>
              <div className="ml-4">
                <h1 className="text-2xl font-bold text-gray-900">Research Documentation</h1>
                <p className="text-sm text-gray-600">MindGuard AI Mental Health Assessment Platform</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant="secondary">Version {researchData.metadata.version}</Badge>
              <Badge variant="outline">{researchData.metadata.review_status.replace('_', ' ')}</Badge>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        {/* Research Overview */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BookOpen className="w-6 h-6" />
              Research Overview
            </CardTitle>
            <CardDescription>
              Comprehensive research documentation for the MindGuard AI-powered mental health assessment platform
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold mb-2">Research Metadata</h4>
                <div className="space-y-2 text-sm">
                  <p><strong>Last Updated:</strong> {researchData.metadata.last_updated}</p>
                  <p><strong>Contributors:</strong> {researchData.metadata.contributors.join(', ')}</p>
                  <p><strong>Review Status:</strong> {researchData.metadata.review_status.replace('_', ' ')}</p>
                </div>
              </div>
              <div>
                <h4 className="font-semibold mb-2">Keywords</h4>
                <div className="flex flex-wrap gap-2">
                  {researchData.metadata.keywords.map((keyword, index) => (
                    <Badge key={index} variant="outline">{keyword}</Badge>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Research Sections */}
        <Tabs defaultValue={researchData.sections[0]?.id} className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-8">
            {researchData.sections.map((section) => (
              <TabsTrigger key={section.id} value={section.id} className="flex items-center gap-2">
                {getSectionIcon(section.id)}
                <span className="hidden sm:inline">{section.title}</span>
              </TabsTrigger>
            ))}
          </TabsList>

          {researchData.sections.map((section) => (
            <TabsContent key={section.id} value={section.id}>
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    {getSectionIcon(section.id)}
                    {section.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="prose max-w-none">
                    <div className="text-gray-700 leading-relaxed mb-6">
                      {section.content.split('\n\n').map((paragraph, index) => (
                        <p key={index} className="mb-4">{paragraph}</p>
                      ))}
                    </div>

                    {section.subsections.length > 0 && (
                      <div>
                        <h3 className="text-lg font-semibold mb-4">Key Areas</h3>
                        <div className="grid gap-4">
                          {section.subsections.map((subsection, index) => (
                            <Card key={index} className="border-l-4 border-l-blue-500">
                              <CardHeader className="pb-2">
                                <CardTitle className="text-base">{subsection.title}</CardTitle>
                              </CardHeader>
                              <CardContent>
                                <p className="text-sm text-gray-600">{subsection.content}</p>
                              </CardContent>
                            </Card>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          ))}
        </Tabs>

        {/* Navigation Footer */}
        <div className="mt-8 flex justify-between items-center">
          <Button variant="outline" asChild>
            <a href="/">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Home
            </a>
          </Button>
          <Button variant="outline" asChild>
            <a href="/dashboard">
              View Dashboard
            </a>
          </Button>
        </div>
      </div>
    </div>
  )
}

