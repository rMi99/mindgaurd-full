"use client"

import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { CheckCircle, Clock } from "lucide-react"

interface MicroAction {
  id: string
  icon: string
  text: string
  reason: string
  duration: string
  category: string
  onClick?: () => void
}

interface MicroActionsProps {
  actions: MicroAction[]
  completedActions: Set<string>
  onActionComplete: (actionId: string) => void
}

export default function MicroActions({ actions, completedActions, onActionComplete }: MicroActionsProps) {
  return (
    <div className="grid gap-4">
      {actions.map((action) => {
        const isCompleted = completedActions.has(action.id)

        return (
          <div
            key={action.id}
            className={`p-4 rounded-lg border-2 transition-all ${
              isCompleted ? "bg-green-50 border-green-200" : "bg-white border-gray-200 hover:border-blue-300"
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="text-2xl">{action.icon}</span>
                <div>
                  <h4 className={`font-medium ${isCompleted ? "text-green-800" : "text-gray-900"}`}>{action.text}</h4>
                  <p className="text-sm text-gray-600">{action.reason}</p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <Badge variant="outline" className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {action.duration}
                </Badge>

                {isCompleted ? (
                  <div className="flex items-center gap-1 text-green-600">
                    <CheckCircle className="w-5 h-5" />
                    <span className="text-sm font-medium">Done</span>
                  </div>
                ) : (
                  <Button
                    size="sm"
                    onClick={() => {
                      if (action.onClick) {
                        action.onClick()
                      }
                      onActionComplete(action.id)
                    }}
                    className="bg-blue-600 hover:bg-blue-700"
                  >
                    Start
                  </Button>
                )}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
