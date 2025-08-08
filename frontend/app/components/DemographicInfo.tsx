"use client"

import type { DemographicData, Language } from "@/lib/types"
import { getTranslation } from "@/lib/translations"

interface DemographicInfoProps {
  data: DemographicData
  onChange: (data: DemographicData) => void
  language: Language
}

export default function DemographicInfo({ data, onChange, language }: DemographicInfoProps) {
  const handleChange = (field: keyof DemographicData, value: string) => {
    onChange({ ...data, [field]: value })
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">{getTranslation(language, "basicInfo")}</h2>
        <p className="text-gray-600">
          Please provide some basic information to help us provide more accurate recommendations.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Age Range</label>
          <select
            value={data.age}
            onChange={(e) => handleChange("age", e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">Select age range</option>
            <option value="18-24">18-24</option>
            <option value="25-34">25-34</option>
            <option value="35-44">35-44</option>
            <option value="45-54">45-54</option>
            <option value="55-64">55-64</option>
            <option value="65+">65+</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Gender</label>
          <select
            value={data.gender}
            onChange={(e) => handleChange("gender", e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">Select gender</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="non-binary">Non-binary</option>
            <option value="prefer-not-to-say">Prefer not to say</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Region/Province</label>
          <select
            value={data.region}
            onChange={(e) => handleChange("region", e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">Select region</option>
            <option value="western">Western Province</option>
            <option value="central">Central Province</option>
            <option value="southern">Southern Province</option>
            <option value="northern">Northern Province</option>
            <option value="eastern">Eastern Province</option>
            <option value="northwest">North Western Province</option>
            <option value="northcentral">North Central Province</option>
            <option value="uva">Uva Province</option>
            <option value="sabaragamuwa">Sabaragamuwa Province</option>
            <option value="other">Other/International</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Education Level</label>
          <select
            value={data.education}
            onChange={(e) => handleChange("education", e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">Select education level</option>
            <option value="primary">Primary Education</option>
            <option value="secondary">Secondary Education</option>
            <option value="diploma">Diploma/Certificate</option>
            <option value="bachelor">Bachelor's Degree</option>
            <option value="master">Master's Degree</option>
            <option value="doctorate">Doctorate</option>
          </select>
        </div>

        <div className="md:col-span-2">
          <label className="block text-sm font-medium text-gray-700 mb-2">Employment Status</label>
          <select
            value={data.employmentStatus}
            onChange={(e) => handleChange("employmentStatus", e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">Select employment status</option>
            <option value="employed-full">Employed (Full-time)</option>
            <option value="employed-part">Employed (Part-time)</option>
            <option value="self-employed">Self-employed</option>
            <option value="unemployed">Unemployed</option>
            <option value="student">Student</option>
            <option value="retired">Retired</option>
            <option value="homemaker">Homemaker</option>
          </select>
        </div>
      </div>
    </div>
  )
}
