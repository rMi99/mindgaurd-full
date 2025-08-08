"use client"

import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Phone, MessageSquare, AlertTriangle } from "lucide-react"

interface EmergencyContact {
  name: string
  number: string
  available24h: boolean
  isText?: boolean
}

interface EmergencyCardProps {
  visible: boolean
  contacts: EmergencyContact[]
}

export default function EmergencyCard({ visible, contacts }: EmergencyCardProps) {
  if (!visible) return null

  const handleCall = (number: string, isText = false) => {
    if (isText) {
      window.open(`sms:${number}`, "_blank")
    } else {
      window.location.href = `tel:${number}`
    }
  }

  return (
    <Card className="border-red-500 bg-red-50">
      <CardContent className="p-6">
        <div className="flex items-center gap-3 mb-4">
          <AlertTriangle className="w-6 h-6 text-red-600" />
          <div>
            <h3 className="font-semibold text-red-800">Crisis Support Available</h3>
            <p className="text-sm text-red-700">
              If you're in crisis or having thoughts of self-harm, please reach out immediately
            </p>
          </div>
        </div>

        <div className="grid gap-3">
          {contacts.map((contact, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-3 bg-white border border-red-200 rounded-lg"
            >
              <div className="flex items-center gap-3">
                {contact.isText ? (
                  <MessageSquare className="w-5 h-5 text-red-600" />
                ) : (
                  <Phone className="w-5 h-5 text-red-600" />
                )}
                <div>
                  <h4 className="font-medium text-gray-900">{contact.name}</h4>
                  <p className="text-sm text-gray-600">{contact.number}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {contact.available24h && (
                  <Badge variant="outline" className="text-green-700 border-green-300">
                    24/7
                  </Badge>
                )}
                <Button
                  onClick={() => handleCall(contact.number, contact.isText)}
                  className="bg-red-600 hover:bg-red-700 text-white"
                  size="sm"
                >
                  {contact.isText ? "Text" : "Call"}
                </Button>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="text-sm text-yellow-800">
            <strong>Remember:</strong> These feelings are temporary. You matter, and help is available.
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
