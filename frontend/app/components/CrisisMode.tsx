"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { AlertTriangle, Phone, Heart, MessageCircle } from "lucide-react"
import type { Language } from "@/lib/types"

interface CrisisModeProps {
  language: Language
  onExit: () => void
}

export default function CrisisMode({ language, onExit }: CrisisModeProps) {
  const [emergencyContacts, setEmergencyContacts] = useState<string[]>([])

  useEffect(() => {
    // Load emergency contacts from localStorage
    const contacts = localStorage.getItem("mindguard_emergency_contacts")
    if (contacts) {
      setEmergencyContacts(JSON.parse(contacts))
    }
  }, [])

  const crisisResources = [
    {
      name: "National Suicide Prevention Lifeline",
      phone: "988",
      description: "24/7 crisis support",
      icon: <Phone className="h-6 w-6" />,
    },
    {
      name: "Crisis Text Line",
      phone: "Text HOME to 741741",
      description: "24/7 text-based crisis support",
      icon: <MessageCircle className="h-6 w-6" />,
    },
    {
      name: "Samaritans Lanka",
      phone: "0717 171 171",
      description: "Free confidential emotional support",
      icon: <Phone className="h-6 w-6" />,
    },
    {
      name: "Emergency Services",
      phone: "911 / 1990",
      description: "Immediate emergency response",
      icon: <AlertTriangle className="h-6 w-6" />,
    },
  ]

  const handleCall = (phoneNumber: string) => {
    // Remove non-numeric characters for tel: link
    const cleanNumber = phoneNumber.replace(/[^\d]/g, "")
    if (cleanNumber) {
      window.location.href = `tel:${cleanNumber}`
    }
  }

  return (
    <div className="min-h-screen bg-red-50 flex items-center justify-center p-4">
      <div className="max-w-2xl w-full space-y-6">
        {/* Header */}
        <Card className="border-red-200 bg-red-50">
          <CardHeader className="text-center">
            <div className="flex justify-center mb-4">
              <div className="p-3 bg-red-100 rounded-full">
                <Heart className="h-8 w-8 text-red-600" />
              </div>
            </div>
            <CardTitle className="text-2xl text-red-800">You Are Not Alone</CardTitle>
            <p className="text-red-700">
              If you're having thoughts of self-harm, please reach out for help immediately. There are people who want
              to support you.
            </p>
          </CardHeader>
        </Card>

        {/* Crisis Resources */}
        <div className="grid gap-4">
          {crisisResources.map((resource, index) => (
            <Card key={index} className="border-red-200">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="p-2 bg-red-100 rounded-lg text-red-600">{resource.icon}</div>
                    <div>
                      <h3 className="font-semibold text-gray-900">{resource.name}</h3>
                      <p className="text-sm text-gray-600">{resource.description}</p>
                    </div>
                  </div>
                  <Button onClick={() => handleCall(resource.phone)} className="bg-red-600 hover:bg-red-700 text-white">
                    {resource.phone}
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Emergency Contacts */}
        {emergencyContacts.length > 0 && (
          <Card className="border-blue-200">
            <CardHeader>
              <CardTitle className="text-blue-800">Your Emergency Contacts</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {emergencyContacts.map((contact, index) => (
                <Button
                  key={index}
                  onClick={() => handleCall(contact)}
                  variant="outline"
                  className="w-full justify-start"
                >
                  <Phone className="h-4 w-4 mr-2" />
                  {contact}
                </Button>
              ))}
            </CardContent>
          </Card>
        )}

        {/* Safety Planning */}
        <Card>
          <CardHeader>
            <CardTitle>Immediate Safety Steps</CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="list-decimal list-inside space-y-2 text-sm">
              <li>Remove any means of self-harm from your immediate environment</li>
              <li>Call one of the crisis numbers above or go to your nearest emergency room</li>
              <li>Stay with someone you trust or ask someone to stay with you</li>
              <li>Avoid alcohol and drugs</li>
              <li>Remember that these feelings are temporary and will pass</li>
            </ol>
          </CardContent>
        </Card>

        {/* Exit Crisis Mode */}
        <div className="text-center">
          <Button onClick={onExit} variant="outline" className="text-gray-600 hover:text-gray-800 bg-transparent">
            I'm feeling safer now - Exit Crisis Mode
          </Button>
        </div>
      </div>
    </div>
  )
}
