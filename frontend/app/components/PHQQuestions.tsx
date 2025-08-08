"use client"

import type { PHQ9Data, Language } from "@/lib/types"
import { getTranslation } from "@/lib/translations"

interface PHQQuestionsProps {
  data: PHQ9Data
  onChange: (data: PHQ9Data) => void
  language: Language
}

const phq9Questions = [
  "Little interest or pleasure in doing things",
  "Feeling down, depressed, or hopeless",
  "Trouble falling or staying asleep, or sleeping too much",
  "Feeling tired or having little energy",
  "Poor appetite or overeating",
  "Feeling bad about yourself or that you are a failure or have let yourself or your family down",
  "Trouble concentrating on things, such as reading the newspaper or watching television",
  "Moving or speaking so slowly that other people could have noticed. Or the opposite being so fidgety or restless that you have been moving around a lot more than usual",
  "Thoughts that you would be better off dead, or of hurting yourself",
]

export default function PHQQuestions({ data, onChange, language }: PHQQuestionsProps) {
  const handleChange = (questionIndex: number, value: number) => {
    onChange({ ...data, [questionIndex + 1]: value })
  }

  const options = [
    { value: 0, label: getTranslation(language, "notAtAll") },
    { value: 1, label: getTranslation(language, "severalDays") },
    { value: 2, label: getTranslation(language, "moreThanHalf") },
    { value: 3, label: getTranslation(language, "nearlyEveryDay") },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">{getTranslation(language, "mentalHealthScreening")}</h2>
        <p className="text-gray-600 mb-6">{getTranslation(language, "overLastTwoWeeks")}</p>
      </div>

      <div className="space-y-8">
        {phq9Questions.map((question, index) => (
          <div key={index} className="bg-gray-50 p-6 rounded-lg">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              {index + 1}. {question}
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {options.map((option) => (
                <label
                  key={option.value}
                  className={`flex items-center p-3 rounded-lg border-2 cursor-pointer transition-all duration-200 ${
                    data[index + 1] === option.value
                      ? "border-blue-500 bg-blue-50 text-blue-700"
                      : "border-gray-200 hover:border-gray-300 hover:bg-gray-50"
                  }`}
                >
                  <input
                    type="radio"
                    name={`question-${index}`}
                    value={option.value}
                    checked={data[index + 1] === option.value}
                    onChange={() => handleChange(index, option.value)}
                    className="sr-only"
                  />
                  <div
                    className={`w-4 h-4 rounded-full border-2 mr-3 flex-shrink-0 ${
                      data[index + 1] === option.value ? "border-blue-500 bg-blue-500" : "border-gray-300"
                    }`}
                  >
                    {data[index + 1] === option.value && (
                      <div className="w-2 h-2 bg-white rounded-full mx-auto mt-0.5"></div>
                    )}
                  </div>
                  <span className="text-sm font-medium">{option.label}</span>
                </label>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Crisis Warning for Question 9 */}
      {data[9] !== null && data[9] > 0 && (
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
              <h3 className="text-sm font-medium text-red-800">
                {getTranslation(language, "immediateSupportAvailable")}
              </h3>
              <div className="mt-2 text-sm text-red-700">
                <p>If you're having thoughts of self-harm, please reach out for help immediately:</p>
                <ul className="mt-2 space-y-1">
                  <li>• Samaritans Lanka: 0717 171 171</li>
                  <li>• Sumithrayo: 0112 682 535</li>
                  <li>• Emergency Services: 1990</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
