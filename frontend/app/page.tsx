"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import type { Language } from "@/lib/types"
import { getTranslation } from "@/lib/translations"
import { useAuthStore } from "@/lib/stores/authStore"
import Header from "./components/Header"
import Footer from "./components/Footer"
import LanguageSelector from "./components/LanguageSelector"
import AssessmentForm from "./components/AssessmentForm"

export default function HomePage() {
  const router = useRouter()
  const { isAuthenticated } = useAuthStore()
  const [language, setLanguage] = useState<Language>("en")
  const [showAssessment, setShowAssessment] = useState(false)

  const handleStartAssessment = () => {
    if (!isAuthenticated) {
      // Store the intended destination and redirect to login
      localStorage.setItem('redirect_after_login', '/assessment')
      router.push('/auth')
      return
    }
    setShowAssessment(true)
  }

  const handleViewDashboard = () => {
    if (!isAuthenticated) {
      // Store the intended destination and redirect to login
      localStorage.setItem('redirect_after_login', '/dashboard')
      router.push('/auth')
      return
    }
    router.push('/dashboard')
  }

  if (showAssessment) {
    return (
      <div className="min-h-screen flex flex-col">
        <Header language={language} />
        <main className="flex-1">
          <AssessmentForm language={language} onLanguageChange={setLanguage} onBack={() => setShowAssessment(false)} />
        </main>
        <Footer language={language} />
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header language={language} />

      <main className="flex-1 flex items-center justify-center px-4 py-12">
        <div className="max-w-4xl mx-auto text-center">
          <div className="mb-8">
            <LanguageSelector currentLanguage={language} onLanguageChange={setLanguage} />
          </div>

          <div className="bg-white rounded-2xl shadow-xl p-8 md:p-12">
            <div className="mb-8">
              <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">{getTranslation(language, "title")}</h1>
              <p className="text-xl text-gray-600 mb-8">{getTranslation(language, "subtitle")}</p>
            </div>

            <div className="bg-blue-50 border-l-4 border-blue-400 p-6 mb-8 text-left">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                    <path
                      fillRule="evenodd"
                      d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-blue-700">{getTranslation(language, "privacyNotice")}</p>
                </div>
              </div>
            </div>

            <div className="grid md:grid-cols-3 gap-6 mb-8">
              <div className="text-center p-6 bg-gray-50 rounded-lg">
                <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">Anonymous & Secure</h3>
                <p className="text-sm text-gray-600">Your privacy is protected with complete anonymity</p>
              </div>

              <div className="text-center p-6 bg-gray-50 rounded-lg">
                <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">AI-Powered</h3>
                <p className="text-sm text-gray-600">Advanced AI provides personalized insights</p>
              </div>

              <div className="text-center p-6 bg-gray-50 rounded-lg">
                <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">Culturally Aware</h3>
                <p className="text-sm text-gray-600">Designed with cultural sensitivity in mind</p>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={handleStartAssessment}
                className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-4 px-8 rounded-lg text-lg transition-colors duration-200 shadow-lg hover:shadow-xl"
              >
                {isAuthenticated ? 'Start Assessment' : 'Login to Start Assessment'}
              </button>
              <button
                onClick={handleViewDashboard}
                className="bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold py-4 px-8 rounded-lg text-lg transition-colors duration-200 border border-gray-300"
              >
                {isAuthenticated ? 'View Dashboard' : 'Login to View Dashboard'}
              </button>
            </div>
          </div>
        </div>
      </main>

      <Footer language={language} />
    </div>
  )
}
