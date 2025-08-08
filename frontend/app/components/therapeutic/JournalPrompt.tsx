"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Save, RefreshCw, Download, Lightbulb } from "lucide-react"

interface JournalPromptProps {
  suggestions: string[]
  riskLevel: string
}

export default function JournalPrompt({ suggestions, riskLevel }: JournalPromptProps) {
  const [currentPrompt, setCurrentPrompt] = useState(0)
  const [journalEntry, setJournalEntry] = useState("")
  const [savedEntries, setSavedEntries] = useState<Array<{ prompt: string; entry: string; date: string }>>([])
  const [wordCount, setWordCount] = useState(0)

  const handleSave = () => {
    if (journalEntry.trim()) {
      const newEntry = {
        prompt: suggestions[currentPrompt],
        entry: journalEntry,
        date: new Date().toLocaleDateString(),
      }
      setSavedEntries((prev) => [newEntry, ...prev])
      setJournalEntry("")
      setWordCount(0)
    }
  }

  const handleNewPrompt = () => {
    setCurrentPrompt((prev) => (prev + 1) % suggestions.length)
    setJournalEntry("")
    setWordCount(0)
  }

  const handleTextChange = (text: string) => {
    setJournalEntry(text)
    const words = text
      .trim()
      .split(/\s+/)
      .filter((word) => word.length > 0)
    setWordCount(words.length)
  }

  const getPromptColor = () => {
    switch (riskLevel) {
      case "high":
        return "bg-red-100 text-red-800 border-red-200"
      case "moderate":
        return "bg-yellow-100 text-yellow-800 border-yellow-200"
      default:
        return "bg-green-100 text-green-800 border-green-200"
    }
  }

  return (
    <div className="space-y-6">
      {/* Current Prompt */}
      <Card className={`border-2 ${getPromptColor()}`}>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Lightbulb className="w-5 h-5" />
              Journal Prompt
            </CardTitle>
            <Badge variant="outline">
              {currentPrompt + 1} of {suggestions.length}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="p-4 bg-white rounded-lg border">
            <p className="text-lg text-gray-800 font-medium">{suggestions[currentPrompt]}</p>
          </div>

          <Textarea
            placeholder="Start writing your thoughts here... There's no right or wrong way to journal."
            value={journalEntry}
            onChange={(e) => handleTextChange(e.target.value)}
            className="min-h-[200px] resize-none"
          />

          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-500">{wordCount} words</span>
            <div className="flex gap-2">
              <Button variant="outline" onClick={handleNewPrompt} className="flex items-center gap-2 bg-transparent">
                <RefreshCw className="w-4 h-4" />
                New Prompt
              </Button>
              <Button onClick={handleSave} disabled={!journalEntry.trim()} className="flex items-center gap-2">
                <Save className="w-4 h-4" />
                Save Entry
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Journaling Benefits */}
      <Card>
        <CardHeader>
          <CardTitle>Benefits of Journaling</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div className="space-y-2">
              <h4 className="font-medium text-gray-900">Mental Health Benefits:</h4>
              <ul className="space-y-1 text-gray-600">
                <li>• Reduces stress and anxiety</li>
                <li>• Improves mood regulation</li>
                <li>• Enhances self-awareness</li>
                <li>• Processes difficult emotions</li>
              </ul>
            </div>
            <div className="space-y-2">
              <h4 className="font-medium text-gray-900">Cognitive Benefits:</h4>
              <ul className="space-y-1 text-gray-600">
                <li>• Clarifies thoughts and feelings</li>
                <li>• Improves problem-solving</li>
                <li>• Boosts memory and comprehension</li>
                <li>• Develops emotional intelligence</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Saved Entries */}
      {savedEntries.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Your Journal Entries</CardTitle>
              <Button variant="outline" size="sm">
                <Download className="w-4 h-4 mr-2" />
                Export All
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4 max-h-60 overflow-y-auto">
              {savedEntries.map((entry, index) => (
                <div key={index} className="p-3 bg-gray-50 rounded-lg border">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-700">{entry.prompt}</span>
                    <span className="text-xs text-gray-500">{entry.date}</span>
                  </div>
                  <p className="text-sm text-gray-600 line-clamp-3">{entry.entry}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
