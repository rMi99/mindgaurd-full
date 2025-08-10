"use client"

import type { AssessmentResult, Language } from "@/lib/types"
import { getTranslation } from "@/lib/translations"

interface ResultsProps {
  results: AssessmentResult
  language: Language
  onStartNew: () => void
}

export default function Results({ results, language, onStartNew }: ResultsProps) {
  const getRiskLevelColor = (riskLevel: string) => {
    switch (riskLevel) {
      case "low":
        return "text-green-700 bg-green-100 border-green-200"
      case "moderate":
        return "text-yellow-700 bg-yellow-100 border-yellow-200"
      case "high":
        return "text-red-700 bg-red-100 border-red-200"
      default:
        return "text-gray-700 bg-gray-100 border-gray-200"
    }
  }

  const getRiskLevelText = (riskLevel: string) => {
    switch (riskLevel) {
      case "low":
        return "Low Risk"
      case "moderate":
        return "Moderate Risk"
      case "high":
        return "High Risk"
      default:
        return "Unknown"
    }
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-8 text-white">
          <h1 className="text-3xl font-bold mb-2">{getTranslation(language, "assessmentComplete")}</h1>
          <p className="text-blue-100">Your personalized mental health assessment results</p>
        </div>

        <div className="p-6 space-y-8">
          {/* Risk Level */}
          <div className="text-center">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">{getTranslation(language, "riskLevel")}</h2>
            <div
              className={`inline-flex items-center px-6 py-3 rounded-full border-2 ${getRiskLevelColor(results.riskLevel)}`}
            >
              <span className="text-2xl font-bold">{getRiskLevelText(results.riskLevel)}</span>
            </div>
            <p className="text-gray-600 mt-2">PHQ-9 Score: {results.phq9Score}/27</p>
            <p className="text-sm text-gray-500 mt-1">Confidence: {Math.round(results.confidenceScore * 100)}%</p>
          </div>

          {/* High Risk Warning */}
          {results.riskLevel === "high" && (
            <div className="bg-red-50 border-l-4 border-red-400 p-6 rounded-lg">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path
                      fillRule="evenodd"
                      d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">Immediate Professional Support Recommended</h3>
                  <p className="mt-2 text-sm text-red-700">
                    Your assessment indicates you may benefit from immediate professional mental health support. Please
                    consider contacting a healthcare provider or crisis helpline.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Emergency Resources */}
          {results.emergencyResources.length > 0 && (
            <div className="bg-blue-50 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Emergency Resources</h3>
              <div className="grid md:grid-cols-2 gap-4">
                {results.emergencyResources.map((resource, index) => (
                  <div key={index} className="bg-white p-4 rounded-lg border">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-gray-900">{resource.name}</h4>
                      {resource.available24h && (
                        <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">24/7</span>
                      )}
                    </div>
                    <p className="text-lg font-mono text-blue-600 mb-2">{resource.phone}</p>
                    <p className="text-sm text-gray-600">{resource.description}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Risk Factors */}
          {results.riskFactors.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Risk Factors Identified</h3>
              <div className="bg-orange-50 rounded-lg p-4">
                <ul className="space-y-2">
                  {results.riskFactors.map((factor, index) => (
                    <li key={index} className="flex items-start">
                      <svg
                        className="h-5 w-5 text-orange-500 mt-0.5 mr-2 flex-shrink-0"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                          clipRule="evenodd"
                        />
                      </svg>
                      <span className="text-sm text-gray-700">{factor}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Protective Factors */}
          {results.protectiveFactors.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Protective Factors</h3>
              <div className="bg-green-50 rounded-lg p-4">
                <ul className="space-y-2">
                  {results.protectiveFactors.map((factor, index) => (
                    <li key={index} className="flex items-start">
                      <svg
                        className="h-5 w-5 text-green-500 mt-0.5 mr-2 flex-shrink-0"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                          clipRule="evenodd"
                        />
                      </svg>
                      <span className="text-sm text-gray-700">{factor}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Recommendations */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Personalized Recommendations</h3>
            <div className="bg-blue-50 rounded-lg p-4">
              <ul className="space-y-3">
                {results.recommendations.map((recommendation, index) => (
                  <li key={index} className="flex items-start">
                    <div className="flex-shrink-0 w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center mr-3 mt-0.5">
                      <span className="text-xs font-medium text-blue-600">{index + 1}</span>
                    </div>
                    <span className="text-sm text-gray-700">{recommendation}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Cultural Considerations */}
          {results.culturalConsiderations.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Cultural Considerations</h3>
              <div className="bg-purple-50 rounded-lg p-4">
                <ul className="space-y-2">
                  {results.culturalConsiderations.map((consideration, index) => (
                    <li key={index} className="flex items-start">
                      <svg
                        className="h-5 w-5 text-purple-500 mt-0.5 mr-2 flex-shrink-0"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                          clipRule="evenodd"
                        />
                      </svg>
                      <span className="text-sm text-gray-700">{consideration}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Disclaimer */}
          <div className="bg-gray-50 rounded-lg p-6 border-l-4 border-gray-400">
            <h4 className="font-medium text-gray-900 mb-2">Important Disclaimer</h4>
            <p className="text-sm text-gray-600">
              This assessment is for screening purposes only and does not constitute a medical diagnosis. The results
              should not replace professional medical advice, diagnosis, or treatment. Always seek the advice of
              qualified healthcare providers with any questions you may have regarding your mental health.
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 pt-6 border-t">
            <button
              onClick={onStartNew}
              className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-6 rounded-lg transition-colors duration-200"
            >
              {getTranslation(language, "startNewAssessment")}
            </button>
            <button
              onClick={() => window.print()}
              className="flex-1 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium py-3 px-6 rounded-lg transition-colors duration-200"
            >
              {getTranslation(language, "viewDetailedReport")}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
