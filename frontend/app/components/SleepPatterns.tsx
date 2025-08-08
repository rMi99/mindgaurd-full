"use client"

import type { SleepData, Language } from "@/lib/types"
import { getTranslation } from "@/lib/translations"

interface SleepPatternsProps {
  data: SleepData
  onChange: (data: SleepData) => void
  language: Language
}

export default function SleepPatterns({ data, onChange, language }: SleepPatternsProps) {
  const handleChange = (field: keyof SleepData, value: string) => {
    onChange({ ...data, [field]: value })
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">{getTranslation(language, "sleepLifestyle")}</h2>
        <p className="text-gray-600">
          These questions help us understand your sleep patterns and lifestyle factors that may affect your mental
          health.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Average hours of sleep per night</label>
          <select
            value={data.sleepHours}
            onChange={(e) => handleChange("sleepHours", e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">Select sleep hours</option>
            <option value="<4">Less than 4 hours</option>
            <option value="4-5">4-5 hours</option>
            <option value="5-6">5-6 hours</option>
            <option value="6-7">6-7 hours</option>
            <option value="7-8">7-8 hours</option>
            <option value="8-9">8-9 hours</option>
            <option value=">9">More than 9 hours</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Sleep quality</label>
          <select
            value={data.sleepQuality}
            onChange={(e) => handleChange("sleepQuality", e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">Select sleep quality</option>
            <option value="excellent">Excellent</option>
            <option value="good">Good</option>
            <option value="fair">Fair</option>
            <option value="poor">Poor</option>
            <option value="very-poor">Very Poor</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Exercise frequency (days per week)</label>
          <select
            value={data.exerciseFrequency}
            onChange={(e) => handleChange("exerciseFrequency", e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">Select exercise frequency</option>
            <option value="never">Never</option>
            <option value="1-2">1-2 days</option>
            <option value="3-4">3-4 days</option>
            <option value="5-6">5-6 days</option>
            <option value="daily">Daily</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Current stress level</label>
          <select
            value={data.stressLevel}
            onChange={(e) => handleChange("stressLevel", e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">Select stress level</option>
            <option value="very-low">Very Low</option>
            <option value="low">Low</option>
            <option value="moderate">Moderate</option>
            <option value="high">High</option>
            <option value="very-high">Very High</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Social support network</label>
          <select
            value={data.socialSupport}
            onChange={(e) => handleChange("socialSupport", e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">Select social support level</option>
            <option value="excellent">Excellent</option>
            <option value="strong">Strong</option>
            <option value="moderate">Moderate</option>
            <option value="minimal">Minimal</option>
            <option value="none">None</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Daily screen time (hours)</label>
          <select
            value={data.screenTime}
            onChange={(e) => handleChange("screenTime", e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">Select screen time</option>
            <option value="<2">Less than 2 hours</option>
            <option value="2-4">2-4 hours</option>
            <option value="4-6">4-6 hours</option>
            <option value="6-8">6-8 hours</option>
            <option value="8-10">8-10 hours</option>
            <option value=">10">More than 10 hours</option>
          </select>
        </div>
      </div>
    </div>
  )
}
