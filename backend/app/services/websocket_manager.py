"""
WebSocket Manager for Real-time Communication
Implements WebSocket connections for live facial analysis updates
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from app.core.patterns import Observer, Subject

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_info: Dict[str, Dict[str, Any]] = {}
        self.message_queue: List[Dict[str, Any]] = []
        self.max_queue_size = 1000
    
    async def connect(self, websocket: WebSocket, client_id: str, user_id: Optional[str] = None) -> None:
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_info[client_id] = {
            'user_id': user_id,
            'connected_at': datetime.now(),
            'last_activity': datetime.now(),
            'message_count': 0
        }
        logger.info(f"WebSocket connected: {client_id} (user: {user_id})")
    
    def disconnect(self, client_id: str) -> None:
        """Remove WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.connection_info[client_id]
            logger.info(f"WebSocket disconnected: {client_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str) -> bool:
        """Send message to specific client"""
        try:
            if client_id in self.active_connections:
                websocket = self.active_connections[client_id]
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(message))
                    self.connection_info[client_id]['last_activity'] = datetime.now()
                    self.connection_info[client_id]['message_count'] += 1
                    return True
        except Exception as e:
            logger.error(f"Error sending personal message to {client_id}: {e}")
            self.disconnect(client_id)
        return False
    
    async def broadcast(self, message: Dict[str, Any], exclude_clients: Optional[Set[str]] = None) -> int:
        """Broadcast message to all connected clients"""
        exclude_clients = exclude_clients or set()
        sent_count = 0
        
        # Create a copy of connections to avoid modification during iteration
        connections_copy = dict(self.active_connections)
        
        for client_id, websocket in connections_copy.items():
            if client_id not in exclude_clients:
                try:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text(json.dumps(message))
                        self.connection_info[client_id]['last_activity'] = datetime.now()
                        self.connection_info[client_id]['message_count'] += 1
                        sent_count += 1
                except Exception as e:
                    logger.error(f"Error broadcasting to {client_id}: {e}")
                    self.disconnect(client_id)
        
        logger.debug(f"Broadcasted message to {sent_count} clients")
        return sent_count
    
    async def broadcast_to_user(self, message: Dict[str, Any], user_id: str) -> int:
        """Broadcast message to all connections of a specific user"""
        sent_count = 0
        
        for client_id, info in self.connection_info.items():
            if info.get('user_id') == user_id:
                if await self.send_personal_message(message, client_id):
                    sent_count += 1
        
        logger.debug(f"Broadcasted message to {sent_count} connections for user {user_id}")
        return sent_count
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        active_count = len(self.active_connections)
        total_messages = sum(info['message_count'] for info in self.connection_info.values())
        
        # Calculate average connection duration
        now = datetime.now()
        durations = []
        for info in self.connection_info.values():
            duration = (now - info['connected_at']).total_seconds()
            durations.append(duration)
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            'active_connections': active_count,
            'total_messages_sent': total_messages,
            'average_connection_duration': avg_duration,
            'queue_size': len(self.message_queue)
        }
    
    def get_user_connections(self, user_id: str) -> List[str]:
        """Get all connection IDs for a user"""
        return [
            client_id for client_id, info in self.connection_info.items()
            if info.get('user_id') == user_id
        ]

class WebSocketObserver(Observer):
    """Observer for WebSocket real-time updates"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.update_count = 0
    
    def update(self, subject: Subject, data: Dict[str, Any]) -> None:
        """Handle updates from observed subjects"""
        try:
            self.update_count += 1
            
            # Create WebSocket message
            message = {
                'type': 'real_time_update',
                'timestamp': datetime.now().isoformat(),
                'update_id': self.update_count,
                'data': data
            }
            
            # Queue message for broadcasting
            asyncio.create_task(self._broadcast_update(message))
            
        except Exception as e:
            logger.error(f"Error in WebSocket observer update: {e}")
    
    async def _broadcast_update(self, message: Dict[str, Any]) -> None:
        """Broadcast update to all connected clients"""
        try:
            await self.connection_manager.broadcast(message)
        except Exception as e:
            logger.error(f"Error broadcasting WebSocket update: {e}")

class FacialAnalysisWebSocketManager:
    """WebSocket manager specifically for facial analysis"""
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.websocket_observer = WebSocketObserver(self.connection_manager)
        self.session_connections: Dict[str, Set[str]] = {}  # session_id -> set of client_ids
    
    async def connect_client(self, websocket: WebSocket, client_id: str, user_id: Optional[str] = None) -> None:
        """Connect a new client"""
        await self.connection_manager.connect(websocket, client_id, user_id)
    
    def disconnect_client(self, client_id: str) -> None:
        """Disconnect a client"""
        # Remove from any sessions
        for session_id, connections in self.session_connections.items():
            connections.discard(client_id)
            if not connections:
                del self.session_connections[session_id]
        
        self.connection_manager.disconnect(client_id)
    
    async def join_session(self, client_id: str, session_id: str) -> bool:
        """Join a client to a session"""
        try:
            if client_id not in self.connection_manager.active_connections:
                return False
            
            if session_id not in self.session_connections:
                self.session_connections[session_id] = set()
            
            self.session_connections[session_id].add(client_id)
            
            # Notify client
            await self.connection_manager.send_personal_message({
                'type': 'session_joined',
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }, client_id)
            
            logger.info(f"Client {client_id} joined session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error joining session: {e}")
            return False
    
    async def leave_session(self, client_id: str, session_id: str) -> bool:
        """Leave a client from a session"""
        try:
            if session_id in self.session_connections:
                self.session_connections[session_id].discard(client_id)
                
                # Notify client
                await self.connection_manager.send_personal_message({
                    'type': 'session_left',
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat()
                }, client_id)
                
                logger.info(f"Client {client_id} left session {session_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error leaving session: {e}")
            return False
    
    async def broadcast_to_session(self, message: Dict[str, Any], session_id: str) -> int:
        """Broadcast message to all clients in a session"""
        if session_id not in self.session_connections:
            return 0
        
        sent_count = 0
        connections = self.session_connections[session_id].copy()
        
        for client_id in connections:
            if await self.connection_manager.send_personal_message(message, client_id):
                sent_count += 1
        
        logger.debug(f"Broadcasted to {sent_count} clients in session {session_id}")
        return sent_count
    
    async def send_analysis_result(self, result: Dict[str, Any], session_id: Optional[str] = None, client_id: Optional[str] = None) -> int:
        """Send facial analysis result"""
        message = {
            'type': 'facial_analysis_result',
            'timestamp': datetime.now().isoformat(),
            'result': result
        }
        
        if client_id:
            # Send to specific client
            if await self.connection_manager.send_personal_message(message, client_id):
                return 1
            return 0
        elif session_id:
            # Send to session
            return await self.broadcast_to_session(message, session_id)
        else:
            # Broadcast to all
            return await self.connection_manager.broadcast(message)
    
    async def send_model_update(self, update: Dict[str, Any]) -> int:
        """Send model update notification"""
        message = {
            'type': 'model_update',
            'timestamp': datetime.now().isoformat(),
            'update': update
        }
        
        return await self.connection_manager.broadcast(message)
    
    async def send_accuracy_alert(self, alert: Dict[str, Any]) -> int:
        """Send accuracy monitoring alert"""
        message = {
            'type': 'accuracy_alert',
            'timestamp': datetime.now().isoformat(),
            'alert': alert
        }
        
        return await self.connection_manager.broadcast(message)
    
    def get_observer(self) -> WebSocketObserver:
        """Get the WebSocket observer for subscribing to updates"""
        return self.websocket_observer
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics"""
        connection_stats = self.connection_manager.get_connection_stats()
        
        return {
            'connection_stats': connection_stats,
            'active_sessions': len(self.session_connections),
            'total_session_connections': sum(len(connections) for connections in self.session_connections.values()),
            'observer_updates': self.websocket_observer.update_count
        }

# Global WebSocket manager instance
websocket_manager = FacialAnalysisWebSocketManager()

def get_websocket_manager() -> FacialAnalysisWebSocketManager:
    """Get the global WebSocket manager instance"""
    return websocket_manager
