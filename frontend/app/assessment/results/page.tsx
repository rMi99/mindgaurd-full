"use client"

import { useEffect, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import Results from '@/app/components/Results'
import type { AssessmentResult, Language } from '@/lib/types'

export default function AssessmentResultsPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [results, setResults] = useState<AssessmentResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [language, setLanguage] = useState<Language>('en')

  useEffect(() => {
    const assessmentId = searchParams.get('assessment_id')
    
    if (!assessmentId) {
      setError('No assessment ID provided')
      setLoading(false)
      return
    }

    // Retrieve the assessment result from localStorage
    const storedResult = localStorage.getItem('assessment_result')
    
    if (!storedResult) {
      setError('Assessment results not found')
      setLoading(false)
      return
    }

    try {
      const parsedResult = JSON.parse(storedResult) as AssessmentResult
      setResults(parsedResult)
    } catch (err) {
      setError('Failed to parse assessment results')
    } finally {
      setLoading(false)
    }
  }, [searchParams])

  const handleStartNew = () => {
    // Clear stored results and redirect to assessment
    localStorage.removeItem('assessment_result')
    router.push('/assessment')
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading your results...</p>
        </div>
      </div>
    )
  }

  if (error || !results) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error || 'Failed to load assessment results'}
          </div>
          <button
            onClick={() => router.push('/assessment')}
            className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded"
          >
            Start New Assessment
          </button>
        </div>
      </div>
    )
  }

  return (
    <Results
      results={results}
      language={language}
      onStartNew={handleStartNew}
    />
  )
} 