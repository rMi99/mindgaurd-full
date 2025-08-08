"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { BookOpen, Headphones, Edit3 } from "lucide-react"
import AudioPlayer from "./AudioPlayer"
import JournalPrompt from "./JournalPrompt"
import GuidedMeditation from "./GuidedMeditation"

interface SelfHelpWidgetsProps {
  riskLevel: "low" | "moderate" | "high"
}

export default function SelfHelpWidgets({ riskLevel }: SelfHelpWidgetsProps) {
  const [activeTab, setActiveTab] = useState("audio")

  const getRecommendedContent = () => {
    switch (riskLevel) {
      case "high":
        return {
          audio: [
            { title: "Crisis Grounding Exercise", duration: "3:00", type: "grounding" },
            { title: "Emergency Calm Down", duration: "5:00", type: "anxiety" },
            { title: "Safe Space Visualization", duration: "8:00", type: "safety" },
          ],
          journal: [
            "What is one thing that feels safe right now?",
            "Name three people you can reach out to",
            "Describe one small step you can take today",
          ],
        }
      case "moderate":
        return {
          audio: [
            { title: "Anxiety Relief Session", duration: "10:00", type: "anxiety" },
            { title: "Mood Boost Meditation", duration: "15:00", type: "mood" },
            { title: "Sleep Preparation", duration: "20:00", type: "sleep" },
          ],
          journal: [
            "What went well today, even if small?",
            "Describe a recent challenge and how you handled it",
            "What are you grateful for right now?",
          ],
        }
      default:
        return {
          audio: [
            { title: "Daily Mindfulness", duration: "10:00", type: "mindfulness" },
            { title: "Energy Boost Session", duration: "12:00", type: "energy" },
            { title: "Evening Wind Down", duration: "25:00", type: "sleep" },
          ],
          journal: [
            "Reflect on your personal growth this week",
            "What are your intentions for tomorrow?",
            "Describe something that inspired you recently",
          ],
        }
    }
  }

  const content = getRecommendedContent()

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Self-Help Therapeutic Tools</h2>
        <p className="text-gray-600">Evidence-based tools personalized for your current mental health status</p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="audio" className="flex items-center gap-2">
            <Headphones className="w-4 h-4" />
            Audio Therapy
          </TabsTrigger>
          <TabsTrigger value="journal" className="flex items-center gap-2">
            <Edit3 className="w-4 h-4" />
            Journaling
          </TabsTrigger>
          <TabsTrigger value="meditation" className="flex items-center gap-2">
            <BookOpen className="w-4 h-4" />
            Guided Practice
          </TabsTrigger>
        </TabsList>

        <TabsContent value="audio" className="space-y-4">
          <AudioPlayer tracks={content.audio} riskLevel={riskLevel} />
        </TabsContent>

        <TabsContent value="journal" className="space-y-4">
          <JournalPrompt suggestions={content.journal} riskLevel={riskLevel} />
        </TabsContent>

        <TabsContent value="meditation" className="space-y-4">
          <GuidedMeditation riskLevel={riskLevel} />
        </TabsContent>
      </Tabs>

      {/* Crisis Support Notice */}
      {riskLevel === "high" && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
              <div>
                <h4 className="font-medium text-red-800">Professional Support Recommended</h4>
                <p className="text-sm text-red-700">
                  While these tools can provide immediate relief, please consider reaching out to a mental health
                  professional for comprehensive support.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
