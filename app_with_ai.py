# Enhanced app.py with AI Agent capabilities
from flask import Flask, request, jsonify, render_template
from datetime import datetime
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import logging

# Import our AI components
try:
    from ai_classifier import SmartTicketAgent
    AI_AVAILABLE = True
    print("🤖 AI Agent mode enabled!")
except ImportError:
    AI_AVAILABLE = False
    print("📋 Rule-based mode (install openai for AI features)")

try:
    from ml_classifier import IntelligentTicketAgent
    ML_AVAILABLE = True
    print("🧠 Machine Learning mode enabled!")
except ImportError:
    ML_AVAILABLE = False
    print("📊 Install scikit-learn for ML features")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@dataclass
class Ticket:
    id: str
    subject: str
    message: str
    customer_email: str
    priority: str
    category: str
    assigned_team: str
    assigned_agent: Optional[str]
    status: str
    created_at: str
    sentiment: str
    estimated_resolution_time: str
    confidence: float = 0.0
    ai_reasoning: str = ""
    suggested_actions: List[str] = None
    needs_escalation: bool = False

class EnhancedTicketRouter:
    def __init__(self):
        # Initialize AI agents based on availability
        self.ai_agent = None
        self.ml_agent = None
        self.mode = "rule-based"
        
        if ML_AVAILABLE:
            try:
                self.ml_agent = IntelligentTicketAgent()
                self.mode = "machine-learning"
                logger.info("🧠 Using Machine Learning Agent")
            except Exception as e:
                logger.warning(f"ML Agent failed to initialize: {e}")
        
        elif AI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            try:
                self.ai_agent = SmartTicketAgent(os.getenv('OPENAI_API_KEY'))
                self.mode = "ai-powered"
                logger.info("🤖 Using OpenAI-powered Agent")
            except Exception as e:
                logger.warning(f"AI Agent failed to initialize: {e}")
        
        # Fallback rule-based system
        self.priority_keywords = {
            'high': ['urgent', 'emergency', 'critical', 'down', 'broken', 'not working', 'crashed', 'error', 'failed'],
            'medium': ['issue', 'problem', 'help', 'support', 'question', 'slow'],
            'low': ['feature', 'request', 'suggestion', 'enhancement', 'feedback']
        }
        
        self.category_keywords = {
            'billing': ['payment', 'invoice', 'bill', 'charge', 'refund', 'subscription', 'pricing', 'cost'],
            'technical': ['bug', 'error', 'crash', 'login', 'password', 'access', 'integration', 'api', 'database'],
            'account': ['profile', 'settings', 'account', 'user', 'permissions', 'registration'],
            'general': ['question', 'info', 'information', 'help', 'support']
        }
        
        self.team_assignments = {
            'billing': 'Finance Team',
            'technical': 'Engineering Team',
            'account': 'Customer Success Team',
            'general': 'General Support Team'
        }
        
        self.agent_assignments = {
            'Finance Team': ['sarah.johnson@company.com', 'mike.chen@company.com'],
            'Engineering Team': ['alex.rodriguez@company.com', 'priya.patel@company.com'],
            'Customer Success Team': ['emma.wilson@company.com', 'david.kim@company.com'],
            'General Support Team': ['lisa.brown@company.com', 'john.smith@company.com']
        }
        
        self.ticket_counter = 1000
    
    def route_ticket(self, subject: str, message: str, customer_email: str) -> Ticket:
        """Enhanced routing with AI capabilities"""
        
        # Try ML Agent first
        if self.ml_agent:
            return self._route_with_ml(subject, message, customer_email)
        
        # Try OpenAI Agent
        elif self.ai_agent:
            return self._route_with_ai(subject, message, customer_email)
        
        # Fallback to rule-based
        else:
            return self._route_with_rules(subject, message, customer_email)
    
    def _route_with_ml(self, subject: str, message: str, customer_email: str) -> Ticket:
        """Route using Machine Learning Agent"""
        try:
            result = self.ml_agent.process_ticket(subject, message, customer_email)
            classification = result['classification']
            
            ticket_id = f"TKT-{self.ticket_counter}"
            self.ticket_counter += 1
            
            return Ticket(
                id=ticket_id,
                subject=subject,
                message=message,
                customer_email=customer_email,
                priority=classification['priority'],
                category=classification['category'],
                assigned_team=result['assigned_team'],
                assigned_agent=self._assign_agent(result['assigned_team']),
                status='open',
                created_at=datetime.now().isoformat(),
                sentiment=classification['sentiment'],
                estimated_resolution_time=result['sla']['resolution_time'],
                confidence=classification['confidence'],
                ai_reasoning=f"ML Classification (Top features: {', '.join(classification['key_features'][:3])})",
                suggested_actions=result['suggested_actions'],
                needs_escalation=result['needs_escalation']
            )
            
        except Exception as e:
            logger.error(f"ML routing failed: {e}")
            return self._route_with_rules(subject, message, customer_email)
    
    def _route_with_ai(self, subject: str, message: str, customer_email: str) -> Ticket:
        """Route using OpenAI Agent"""
        try:
            result = self.ai_agent.process_ticket(subject, message, customer_email)
            classification = result['classification']
            
            ticket_id = f"TKT-{self.ticket_counter}"
            self.ticket_counter += 1
            
            return Ticket(
                id=ticket_id,
                subject=subject,
                message=message,
                customer_email=customer_email,
                priority=classification.priority,
                category=classification.category,
                assigned_team=result['assigned_team'],
                assigned_agent=self._assign_agent(result['assigned_team']),
                status='open',
                created_at=datetime.now().isoformat(),
                sentiment=classification.sentiment,
                estimated_resolution_time=result['estimated_resolution_time'],
                confidence=classification.confidence,
                ai_reasoning=classification.reasoning,
                suggested_actions=classification.suggested_actions,
                needs_escalation=result.get('escalation_needed', False)
            )
            
        except Exception as e:
            logger.error(f"AI routing failed: {e}")
            return self._route_with_rules(subject, message, customer_email)
    
    def _route_with_rules(self, subject: str, message: str, customer_email: str) -> Ticket:
        """Fallback rule-based routing"""
        combined_text = f"{subject} {message}"
        
        priority = self._analyze_priority(combined_text)
        category = self._analyze_category(combined_text)
        sentiment = self._analyze_sentiment(combined_text)
        assigned_team = self.team_assignments[category]
        assigned_agent = self._assign_agent(assigned_team)
        estimated_time = self._estimate_resolution_time(priority, category)
        
        ticket_id = f"TKT-{self.ticket_counter}"
        self.ticket_counter += 1
        
        return Ticket(
            id=ticket_id,
            subject=subject,
            message=message,
            customer_email=customer_email,
            priority=priority,
            category=category,
            assigned_team=assigned_team,
            assigned_agent=assigned_agent,
            status='open',
            created_at=datetime.now().isoformat(),
            sentiment=sentiment,
            estimated_resolution_time=estimated_time,
            confidence=75.0,
            ai_reasoning="Rule-based classification",
            suggested_actions=["Route to team", "Send acknowledgment"],
            needs_escalation=priority == 'high'
        )
    
    def _analyze_priority(self, text: str) -> str:
        """Rule-based priority analysis"""
        text_lower = text.lower()
        
        high_score = sum(1 for keyword in self.priority_keywords['high'] if keyword in text_lower)
        medium_score = sum(1 for keyword in self.priority_keywords['medium'] if keyword in text_lower)
        low_score = sum(1 for keyword in self.priority_keywords['low'] if keyword in text_lower)
        
        if any(word in text_lower for word in ['asap', 'immediately', 'urgent', 'emergency']):
            return 'high'
        
        if high_score >= 2 or high_score > medium_score:
            return 'high'
        elif medium_score > 0 or medium_score > low_score:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_category(self, text: str) -> str:
        """Rule-based category analysis"""
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category] = score
        
        if max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        return 'general'
    
    def _analyze_sentiment(self, text: str) -> str:
        """Rule-based sentiment analysis"""
        text_lower = text.lower()
        
        negative_words = ['angry', 'frustrated', 'terrible', 'awful', 'hate', 'disgusted', 'disappointed']
        positive_words = ['great', 'excellent', 'love', 'amazing', 'wonderful', 'fantastic']
        
        negative_score = sum(1 for word in negative_words if word in text_lower)
        positive_score = sum(1 for word in positive_words if word in text_lower)
        
        if negative_score > positive_score:
            return 'negative'
        elif positive_score > negative_score:
            return 'positive'
        else:
            return 'neutral'
    
    def _estimate_resolution_time(self, priority: str, category: str) -> str:
        """Estimate resolution time"""
        base_times = {
            'billing': {'high': '2 hours', 'medium': '4 hours', 'low': '1 day'},
            'technical': {'high': '4 hours', 'medium': '8 hours', 'low': '2 days'},
            'account': {'high': '1 hour', 'medium': '2 hours', 'low': '4 hours'},
            'general': {'high': '2 hours', 'medium': '6 hours', 'low': '1 day'}
        }
        
        return base_times.get(category, {}).get(priority, '1 day')
    
    def _assign_agent(self, team: str) -> str:
        """Simple round-robin agent assignment"""
        agents = self.agent_assignments.get(team, ['support@company.com'])
        return agents[self.ticket_counter % len(agents)]
    
    def get_system_status(self) -> Dict:
        """Get current system capabilities"""
        return {
            'mode': self.mode,
            'ai_available': AI_AVAILABLE,
            'ml_available': ML_AVAILABLE,
            'openai_configured': bool(os.getenv('OPENAI_API_KEY')),
            'features': {
                'intelligent_classification': self.mode != 'rule-based',
                'confidence_scoring': True,
                'sentiment_analysis': True,
                'escalation_detection': True,
                'action_suggestions': self.mode != 'rule-based'
            }
        }

# Initialize the enhanced router
router = EnhancedTicketRouter()
tickets_db = []

@app.route('/')
def index():
    """Render the enhanced dashboard"""
    return render_template('index.html')

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get system capabilities and status"""
    return jsonify(router.get_system_status())

@app.route('/api/tickets', methods=['POST'])
def create_ticket():
    """Create a new support ticket with AI processing"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['subject', 'message', 'customer_email']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Route the ticket with AI
        ticket = router.route_ticket(
            subject=data['subject'],
            message=data['message'],
            customer_email=data['customer_email']
        )
        
        # Store ticket
        tickets_db.append(asdict(ticket))
        
        # Enhanced response with AI insights
        response = {
            'success': True,
            'ticket': asdict(ticket),
            'message': f'Ticket {ticket.id} created and routed to {ticket.assigned_team}',
            'ai_insights': {
                'mode': router.mode,
                'confidence': ticket.confidence,
                'reasoning': ticket.ai_reasoning,
                'escalation_needed': ticket.needs_escalation
            }
        }
        
        if ticket.suggested_actions:
            response['ai_insights']['suggested_actions'] = ticket.suggested_actions
        
        return jsonify(response), 201
        
    except Exception as e:
        logger.error(f"Error creating ticket: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Keep all other existing endpoints from the original app.py
@app.route('/api/tickets', methods=['GET'])
def get_tickets():
    """Get all tickets with optional filtering"""
    try:
        priority = request.args.get('priority')
        category = request.args.get('category')
        status = request.args.get('status')
        
        filtered_tickets = tickets_db.copy()
        
        if priority:
            filtered_tickets = [t for t in filtered_tickets if t['priority'] == priority]
        if category:
            filtered_tickets = [t for t in filtered_tickets if t['category'] == category]
        if status:
            filtered_tickets = [t for t in filtered_tickets if t['status'] == status]
        
        return jsonify({
            'tickets': filtered_tickets,
            'total': len(filtered_tickets)
        })
        
    except Exception as e:
        logger.error(f"Error fetching tickets: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get enhanced analytics with AI insights"""
    try:
        if not tickets_db:
            return jsonify({
                'total_tickets': 0,
                'by_priority': {},
                'by_category': {},
                'by_status': {},
                'by_sentiment': {},
                'ai_metrics': {
                    'average_confidence': 0,
                    'escalation_rate': 0,
                    'mode': router.mode
                }
            })
        
        # Calculate basic metrics
        total_tickets = len(tickets_db)
        
        by_priority = {}
        by_category = {}
        by_status = {}
        by_sentiment = {}
        confidence_scores = []
        escalation_count = 0
        
        for ticket in tickets_db:
            # Basic breakdowns
            by_priority[ticket['priority']] = by_priority.get(ticket['priority'], 0) + 1
            by_category[ticket['category']] = by_category.get(ticket['category'], 0) + 1
            by_status[ticket['status']] = by_status.get(ticket['status'], 0) + 1
            by_sentiment[ticket['sentiment']] = by_sentiment.get(ticket['sentiment'], 0) + 1
            
            # AI metrics
            confidence_scores.append(ticket.get('confidence', 0))
            if ticket.get('needs_escalation', False):
                escalation_count += 1
        
        return jsonify({
            'total_tickets': total_tickets,
            'by_priority': by_priority,
            'by_category': by_category,
            'by_status': by_status,
            'by_sentiment': by_sentiment,
            'ai_metrics': {
                'average_confidence': round(sum(confidence_scores) / len(confidence_scores), 1),
                'escalation_rate': round((escalation_count / total_tickets) * 100, 1),
                'mode': router.mode
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating analytics: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with AI status"""
    status = router.get_system_status()
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'ai_mode': status['mode'],
        'capabilities': status['features']
    })

if __name__ == '__main__':
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Show system status
    status = router.get_system_status()
    print(f"\n🎯 System Status:")
    print(f"   Mode: {status['mode']}")
    print(f"   AI Available: {status['ai_available']}")
    print(f"   ML Available: {status['ml_available']}")
    print(f"   OpenAI Configured: {status['openai_configured']}")
    
    # Create sample tickets
    sample_tickets = [
        {
            'subject': 'URGENT: Cannot access my account',
            'message': 'I keep getting an error when trying to log in. This is urgent as I need to access my files for a client presentation!',
            'customer_email': 'john.doe@example.com'
        },
        {
            'subject': 'Billing question about subscription',
            'message': 'I was charged twice this month for my subscription. Can you please help?',
            'customer_email': 'jane.smith@example.com'
        },
        {
            'subject': 'Love the new features!',
            'message': 'The new dashboard looks amazing! It would be great if you could add dark mode to the application.',
            'customer_email': 'tech.user@example.com'
        }
    ]
    
    # Create sample tickets with AI processing
    for sample in sample_tickets:
        ticket = router.route_ticket(
            subject=sample['subject'],
            message=sample['message'],
            customer_email=sample['customer_email']
        )
        tickets_db.append(asdict(ticket))
        print(f"   Created {ticket.id}: {ticket.priority}/{ticket.category} (Confidence: {ticket.confidence}%)")
    
    app.run(debug=True, host='0.0.0.0', port=5000)