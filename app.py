from flask import Flask, request, jsonify, render_template
from datetime import datetime
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import logging

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

class TicketRouter:
    def __init__(self):
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
        
        # Simple ticket counter for ID generation
        self.ticket_counter = 1000
    
    def analyze_priority(self, text: str) -> str:
        """Analyze text to determine priority level"""
        text_lower = text.lower()
        
        # High priority indicators
        high_score = sum(1 for keyword in self.priority_keywords['high'] if keyword in text_lower)
        medium_score = sum(1 for keyword in self.priority_keywords['medium'] if keyword in text_lower)
        low_score = sum(1 for keyword in self.priority_keywords['low'] if keyword in text_lower)
        
        # Check for urgency indicators
        if any(word in text_lower for word in ['asap', 'immediately', 'urgent', 'emergency']):
            return 'high'
        
        if high_score >= 2 or high_score > medium_score:
            return 'high'
        elif medium_score > 0 or medium_score > low_score:
            return 'medium'
        else:
            return 'low'
    
    def analyze_category(self, text: str) -> str:
        """Analyze text to determine category"""
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category] = score
        
        # Return category with highest score, default to 'general'
        if max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        return 'general'
    
    def analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis"""
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
    
    def estimate_resolution_time(self, priority: str, category: str) -> str:
        """Estimate resolution time based on priority and category"""
        base_times = {
            'billing': {'high': '2 hours', 'medium': '4 hours', 'low': '1 day'},
            'technical': {'high': '4 hours', 'medium': '8 hours', 'low': '2 days'},
            'account': {'high': '1 hour', 'medium': '2 hours', 'low': '4 hours'},
            'general': {'high': '2 hours', 'medium': '6 hours', 'low': '1 day'}
        }
        
        return base_times.get(category, {}).get(priority, '1 day')
    
    def assign_agent(self, team: str) -> str:
        """Simple round-robin agent assignment"""
        agents = self.agent_assignments.get(team, ['support@company.com'])
        # In a real system, this would use a more sophisticated load balancing
        return agents[self.ticket_counter % len(agents)]
    
    def route_ticket(self, subject: str, message: str, customer_email: str) -> Ticket:
        """Main routing logic"""
        combined_text = f"{subject} {message}"
        
        priority = self.analyze_priority(combined_text)
        category = self.analyze_category(combined_text)
        sentiment = self.analyze_sentiment(combined_text)
        assigned_team = self.team_assignments[category]
        assigned_agent = self.assign_agent(assigned_team)
        estimated_time = self.estimate_resolution_time(priority, category)
        
        # Generate ticket ID
        ticket_id = f"TKT-{self.ticket_counter}"
        self.ticket_counter += 1
        
        ticket = Ticket(
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
            estimated_resolution_time=estimated_time
        )
        
        logger.info(f"Created ticket {ticket_id} - Priority: {priority}, Category: {category}")
        return ticket

# Initialize the router
router = TicketRouter()

# In-memory storage (in production, use a database)
tickets_db = []

@app.route('/')
def index():
    """Render the main dashboard"""
    return render_template('index.html')

@app.route('/api/tickets', methods=['POST'])
def create_ticket():
    """Create a new support ticket"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['subject', 'message', 'customer_email']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Route the ticket
        ticket = router.route_ticket(
            subject=data['subject'],
            message=data['message'],
            customer_email=data['customer_email']
        )
        
        # Store ticket (in production, save to database)
        tickets_db.append(asdict(ticket))
        
        return jsonify({
            'success': True,
            'ticket': asdict(ticket),
            'message': f'Ticket {ticket.id} created and routed to {ticket.assigned_team}'
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating ticket: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/tickets', methods=['GET'])
def get_tickets():
    """Get all tickets with optional filtering"""
    try:
        # Get query parameters for filtering
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

@app.route('/api/tickets/<ticket_id>', methods=['GET'])
def get_ticket(ticket_id):
    """Get a specific ticket by ID"""
    try:
        ticket = next((t for t in tickets_db if t['id'] == ticket_id), None)
        if not ticket:
            return jsonify({'error': 'Ticket not found'}), 404
        
        return jsonify({'ticket': ticket})
        
    except Exception as e:
        logger.error(f"Error fetching ticket {ticket_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/tickets/<ticket_id>/status', methods=['PUT'])
def update_ticket_status(ticket_id):
    """Update ticket status"""
    try:
        data = request.get_json()
        new_status = data.get('status')
        
        if new_status not in ['open', 'in_progress', 'resolved', 'closed']:
            return jsonify({'error': 'Invalid status'}), 400
        
        ticket = next((t for t in tickets_db if t['id'] == ticket_id), None)
        if not ticket:
            return jsonify({'error': 'Ticket not found'}), 404
        
        ticket['status'] = new_status
        
        return jsonify({
            'success': True,
            'message': f'Ticket {ticket_id} status updated to {new_status}'
        })
        
    except Exception as e:
        logger.error(f"Error updating ticket {ticket_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get basic analytics about tickets"""
    try:
        if not tickets_db:
            return jsonify({
                'total_tickets': 0,
                'by_priority': {},
                'by_category': {},
                'by_status': {},
                'by_sentiment': {}
            })
        
        # Calculate metrics
        total_tickets = len(tickets_db)
        
        by_priority = {}
        by_category = {}
        by_status = {}
        by_sentiment = {}
        
        for ticket in tickets_db:
            # Priority breakdown
            priority = ticket['priority']
            by_priority[priority] = by_priority.get(priority, 0) + 1
            
            # Category breakdown
            category = ticket['category']
            by_category[category] = by_category.get(category, 0) + 1
            
            # Status breakdown
            status = ticket['status']
            by_status[status] = by_status.get(status, 0) + 1
            
            # Sentiment breakdown
            sentiment = ticket['sentiment']
            by_sentiment[sentiment] = by_sentiment.get(sentiment, 0) + 1
        
        return jsonify({
            'total_tickets': total_tickets,
            'by_priority': by_priority,
            'by_category': by_category,
            'by_status': by_status,
            'by_sentiment': by_sentiment
        })
        
    except Exception as e:
        logger.error(f"Error generating analytics: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    # Create templates directory and sample data for demo
    os.makedirs('templates', exist_ok=True)
    
    # Load sample tickets for demo
    sample_tickets = [
        {
            'subject': 'Cannot access my account',
            'message': 'I keep getting an error when trying to log in. This is urgent as I need to access my files.',
            'customer_email': 'john.doe@example.com'
        },
        {
            'subject': 'Billing question about subscription',
            'message': 'I was charged twice this month for my subscription. Can you please help?',
            'customer_email': 'jane.smith@example.com'
        },
        {
            'subject': 'Feature request',
            'message': 'It would be great if you could add dark mode to the application.',
            'customer_email': 'tech.user@example.com'
        }
    ]
    
    # Create sample tickets
    for sample in sample_tickets:
        ticket = router.route_ticket(
            subject=sample['subject'],
            message=sample['message'],
            customer_email=sample['customer_email']
        )
        tickets_db.append(asdict(ticket))
    
    app.run(debug=True, host='0.0.0.0', port=5000)